import pandas as pd
import uuid
import re
import requests
import sys
import os

from collections import namedtuple

from typing import List, Tuple, Dict, Any

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from rouge import Rouge
import spacy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))

from config import get_config

import optuna
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ParamsSelector:
    _EncoderInfo = namedtuple('EncoderInfo', ['bi_encoder', 'bi_encoder_dim'])
    _TestResult = namedtuple('TestResult', ['result', 'score'])
    _SearchResult = namedtuple('SearchResult', ['top_chunks', 'top_files'])

    def __init__(self, model_name: str, df: pd.DataFrame, n_trials: int) -> None:
        self.model_name = model_name
        self.df = df
        self.n_trials = n_trials
        self._config = get_config()
        self._qdrant_client = QdrantClient(self._config['database']['path_bd'], api_key=self._config['database']['api_key'])
        self._lemmatizer = spacy.load('ru_core_news_sm')
        self.best_score = 0
        self.best_result = None
        self.study = None

    def _file_to_chunks(self, file_name: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        file_path = f'{self._config['text_folder_path']}/{file_name}'
        loader = TextLoader(file_path, encoding='utf-8')
        document = loader.load()
        content = document[0].page_content

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len,
            is_separator_regex = False,
            add_start_index = False
        )

        chunks = text_splitter.split_text(content)

        return chunks
    
    def _get_bi_encoder(self, encoder_name: str) -> tuple[SentenceTransformer, int]:
        raw_model = Transformer(encoder_name)

        bi_encoder_dim = raw_model.get_word_embedding_dimension()

        pooling_model = Pooling(
            bi_encoder_dim, 
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
        )

        bi_encoder = SentenceTransformer(
            modules=[raw_model, pooling_model], 
        )
        return self._EncoderInfo(bi_encoder, bi_encoder_dim)

    @staticmethod
    def _string_to_vector(bi_encoder: SentenceTransformer, text: str) -> List[float]:
        embeddings = bi_encoder.encode(
            text, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embeddings
    
    def _save_chunks(self, bi_encoder: SentenceTransformer, chunks: List[str], file_name: str) -> Dict[str, Any]:
        chunk_embeddings = self._string_to_vector(bi_encoder, chunks)

        points = []
        for i in range(len(chunk_embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector = chunk_embeddings[i], 
                payload={'file': file_name, 'chunk': chunks[i]}
            )
            points.append(point)
        
        operation_info = self._qdrant_client.upsert(
            collection_name = self._config['database']['coll_name'],
            wait = True,
            points = points
        )
        
        return operation_info

    def _files_to_vecdb(self, files: List[str], bi_encoder: SentenceTransformer, vec_size: int, chunk_size: int, chunk_overlap: int) -> None:    
        self._qdrant_client.delete_collection(collection_name=self._config['database']['coll_name'])
        self._qdrant_client.create_collection(
            collection_name = self._config['database']['coll_name'],
            vectors_config = VectorParams(size=vec_size, distance=Distance.COSINE),
        )
        
        for file_name in files:
            chunks = self._file_to_chunks(file_name, chunk_size, chunk_overlap)
            operation_status = self._save_chunks(bi_encoder, chunks, file_name)
    
    def _vec_search(self, bi_encoder: SentenceTransformer, query: str, n_top_cos: int) -> Tuple[List[str], List[str]]:
        query_embeddings = self._string_to_vector(bi_encoder, query)

        search_results = self._qdrant_client.search(
            collection_name = self._config['database']['coll_name'],
            query_vector = query_embeddings,
            limit = n_top_cos,
            with_vectors = False
        )

        top_chunks = [x.payload['chunk'] for x in search_results]
        top_files = list(set([x.payload['file'] for x in search_results]))

        return self._SearchResult(top_chunks, top_files)
    
    def _get_llm_answer(self, query: str, chunks_join: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        user_prompt = '''Используй только следующий контекст, чтобы очень кратко ответить на вопрос в конце.
        Не пытайся выдумывать ответ.
        Контекст:
        ===========  
        {chunks_join}  
        ===========  
        Вопрос:  
        ===========  
        {query}'''.format(chunks_join=chunks_join, query=query)

        system_prompt = "Ты — Олег, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. Отвечай только строго по контексту одним предложением."

        url = self._config['url']
        headers = {
        "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": top_p
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Ошибка запроса: {response.status_code}, {response.text}")
    
    def _lemmatize(self, string: str) -> str:
        clear = re.sub(self._config['patterns'], ' ', string)
        tokens = []
        for token in clear.split():
            if token:
                token = token.strip()
                doc = self._lemmatizer(token)
                lemmatized_words = [token.lemma_ for token in doc]
                token  = " ".join(lemmatized_words)
                tokens.append(token)

        tokens = ' '.join(tokens)
        return tokens
    
    def get_llm_score(self, answer: str, answer_true: str) -> float:
        answer = self._lemmatize(answer)
        answer_true = self._lemmatize(answer_true)
        if len(answer) == 0:
            answer = '-'
    
        rouge = Rouge()
        scores = rouge.get_scores(answer, answer_true)[0]
        rouge_1 = round(scores['rouge-1']['r']*100, 2)
        
        return rouge_1
    
    @staticmethod
    def get_context_score(chunks_join: str, context: str) -> float:
        rouge = Rouge()
        scores = rouge.get_scores(chunks_join, context)[0]
        score = round(scores['rouge-l']['r'] * 100)
        
        return score

    def test(self, encoder_name: str, chunk_size: int, chunk_overlap: int, n_top_cos: int, max_new_tokens: int, temperature: float, top_p: float) -> Tuple[pd.DataFrame, float]:
        try:
            bi_encoder_info = self._get_bi_encoder(encoder_name)
            files = self.df['Файл'].unique()
            self._files_to_vecdb(files, bi_encoder_info.bi_encoder, bi_encoder_info.bi_encoder_dim, chunk_size, chunk_overlap)
            result = []
            for i, row in self.df.iterrows():
                query = row['Вопрос']
                answer_true = row['Правильный ответ']
                file_name = row['Файл']
                context = row['Контекст']

                search_result = self._vec_search(bi_encoder_info.bi_encoder, query, n_top_cos)
                row['top_files'] = search_result.top_files
                row['top_chunks'] = search_result.top_chunks
                top_chunks_join = '\n'.join(search_result.top_chunks)

                answer = self._get_llm_answer(query, top_chunks_join, max_new_tokens, temperature, top_p)
                row['Ответ'] = answer

                row['file_score'] = int(file_name in search_result.top_files)
                row['context_score'] = self.get_context_score(top_chunks_join, context)
                row['llm_score'] = self.get_llm_score(answer, answer_true)

                result.append(row)

            result = pd.DataFrame(result)
            result = result.sort_values(by=['llm_score','context_score','file_score'], ascending=False)
            result = result.reset_index(drop=True)

            score = result['llm_score'].mean()

            return self._TestResult(result, score)
        
        except Exception as e:
            print(e)
            return self._TestResult(None, 0)

    def _objective(self, trial) -> float:
        encoder_name = trial.suggest_categorical('encoder_name', self._config['optuna_params']['encoder_name'])
        chunk_size = trial.suggest_int('chunk_size', self._config['optuna_params']['chunk_size_min'], self._config['optuna_params']['chunk_size_max'])
        chunk_overlap = trial.suggest_int('chunk_overlap', self._config['optuna_params']['chunk_overlap_min'], self._config['optuna_params']['chunk_overlap_max'])
        n_top_cos = trial.suggest_int('n_top_cos', self._config['optuna_params']['n_top_cos_min'], self._config['optuna_params']['n_top_cos_max'])
        max_new_tokens = trial.suggest_int('max_new_tokens', self._config['optuna_params']['max_new_tokens_min'], self._config['optuna_params']['max_new_tokens_max'])
        temperature = trial.suggest_float('temperature', self._config['optuna_params']['temperature_min'], self._config['optuna_params']['temperature_max'])
        top_p = trial.suggest_float('top_p', self._config['optuna_params']['top_p_min'], self._config['optuna_params']['top_p_max'])

        test_result = self.test(
            encoder_name=encoder_name,
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            n_top_cos=n_top_cos,
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_p=top_p
        )

        if test_result.score > self.best_score:
            self.best_score = test_result.score
            best_score_tag = '<---'
            self.best_result = test_result.result
        else:
            best_score_tag = ''
        
        print(f'{test_result.score:.2f}', best_score_tag)

        return test_result.score
    
    def start_optuna(self) -> None:
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)
        self.best_result.to_pickle(f'results_for_{self.model_name}_{self.n_trials}_trials.pkl')