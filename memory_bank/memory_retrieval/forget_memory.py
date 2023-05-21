import json, datetime
import random, copy
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document
from typing import List, Tuple, Optional
from langchain.vectorstores import FAISS
from memory_retrieval.textsplitter import ChineseTextSplitter
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from memory_retrieval.configs.model_config import *
# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 6

# LLM input history length
LLM_HISTORY_LEN = 3
import math

def forgetting_curve(t, S):
    """
    Calculate the retention of information at time t based on the forgetting curve.

    :param t: Time elapsed since the information was learned (in days).
    :type t: float
    :param S: Strength of the memory.
    :type S: float
    :return: Retention of information at time t.
    :rtype: float
    Memory strength is a concept used in memory models to represent the durability or stability of a memory trace in the brain. 
    In the context of the forgetting curve, memory strength (denoted as 'S') is a parameter that 
    influences the rate at which information is forgotten. 
    The higher the memory strength, the slower the rate of forgetting, 
    and the longer the information is retained.
    """
    return math.exp(-t / S)

    # # Example usage
    # t = 1  # Time elapsed since the information was learned (in days)
    # S = 7  # Strength of the memory

    # retention = forgetting_curve(t, S)
    # print("Retention after", t, "day(s):", retention)

class MemoryForgetterLoader(UnstructuredFileLoader):
    def __init__(self, filepath,language,mode="elements"):
        super().__init__(filepath, mode=mode)
        self.filepath = filepath
        self.memory_level = 3
        self.total_date = 30
        self.language = language
        self.forget_rate = \
        [dict([(i, forgetting_curve(i,1)) for i in range(self.total_date)]),
        dict([(i, forgetting_curve(i,2)) for i in range(self.total_date)]),
        dict([(i, forgetting_curve(i,3)) for i in range(self.total_date)])]
        self.memory_bank = {}
        self.user_memory = {'history':dict([(f'level_{i}', []) for i in range(self.memory_level)]),
                            'summary':{},
                            'overall_history':'',
                            'overall_personality':''}

    def _get_metadata(self, date: str) -> dict:
        return {"source": date,'cur_level_date':date}
    
    def _get_date_difference(self, date1: str, date2: str) -> int:
        date_format = "%Y-%m-%d"
        d1 = datetime.datetime.strptime(date1, date_format)
        d2 = datetime.datetime.strptime(date2, date_format)
        return (d2 - d1).days

    def forget_memory(self, cur_date):
        forget_f = open(self.file_path.replace('.json' , '_forget_context.txt'), 'w', encoding='utf-8')
        for user in self.memory_bank.keys():
            for i in range(self.memory_level):
                if len(self.memory_bank[user]['history'][f'level_{i}']) == 0:
                    continue
                # Create a new list to store the memories after forgetting
                new_memory_level = []
                for memory in self.memory_bank[user]['history'][f'level_{i}']:
                    source_date = memory['metadata']['cur_level_date']
                    days_diff = self._get_date_difference(source_date, cur_date)
                    # Check if the memory's age is within the range of the forget_rate dictionary
                    if days_diff < self.total_date:
                        # Calculate the probability of retaining the memory
                        retention_probability = self.forget_rate[i].get(days_diff, 0)
                        # Keep the memory with the retention_probability
                        if random.random() < retention_probability:
                            new_memory_level.append(memory)
                        else:
                            
                            print('{user} Forgetted memory in {cur_date} : memory {memory}\n'.format(user=user,cur_date=cur_date,memory=memory['metadata']['source']))
                            forget_f.write('{user} Forgetted memory in {cur_date} : memory {memory}\n'.format(user=user,cur_date=cur_date,memory=memory['metadata']['source']))
                # Replace the current memory bank level with the new_memory_level
                self.memory_bank[user]['history'][f'level_{i}'] = new_memory_level
     
    def update_memory_when_searched(self, recalled,user,cur_date):
        recalled_date = recalled.metadata['source']
        recalled_level = recalled.metadata['memory_level']
        # Find the memory in the memory_bank and its current level
        recalled_memory = None
        current_level = -1
        memory_index = -1
        for i,memory in enumerate(self.memory_bank[user]['history'][f'level_{recalled_level}']):
            if memory['metadata']['source'] == recalled_date:
                recalled_memory = copy.deepcopy(memory)
                current_level = recalled_level
                memory_index = i
                break
        
        # If the memory is not found, return
        if current_level == -1:
            return

        # Remove the memory from the current level
        self.memory_bank[user]['history'][f'level_{current_level}'].pop(memory_index)
        # self.memory_bank[user]['history'][f'level_{current_level}'].remove(recalled_memory)

        # Update the memory's metadata with the new date
        recalled_memory['metadata']['cur_level_date'] = cur_date

        # Move the memory to a higher level in the memory_bank, if possible
        new_level = current_level + 1 if current_level < self.memory_level-1 else self.memory_level-1
        self.memory_bank[user]['history'][f'level_{new_level}'].append(recalled_memory)
    
    def write_memories(self, out_file):
        with open(out_file, "w", encoding="utf-8") as f:
            print(f'Successfully write to {out_file}')
            json.dump(self.memory_bank, f, ensure_ascii=False, indent=4)
    
    def load_memories(self, memory_file):
        # print(memory_file)
        with open(memory_file, "r", encoding="utf-8") as f:
            self.memory_bank = json.load(f)

    def get_docs(self,name):
        docs = []
        for i in range(self.memory_level):
            for doc in self.memory_bank[name]['history'][f'level_{i}']:
                doc['metadata']['memory_level'] = i
                docs.append(Document(**doc))
        # print(docs)
        # exit()
        return docs

    def load_update_and_split(
        self, text_splitter: Optional[TextSplitter] = None,name='',cur_date=''
    ) -> List[Document]:
        """Load documents and split into chunks."""
        if text_splitter is None:
            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            _text_splitter = text_splitter
        if not cur_date:
            cur_date = datetime.date.today().strftime("%Y-%m-%d")
        # self.memory_bank = json.load(open(self.filepath.replace('.json' , '_forget_format.json'), "r", encoding="utf-8"))
        # self.load_memories(self.filepath.replace('.json' , '_forget_format.json'))
        self.forget_memory(cur_date=cur_date)
        docs = self.get_docs(name=name)
        results = _text_splitter.split_documents(docs)
        # print(results)
        return results
    
    def initial_load_and_save(self,name):
        with open(self.filepath, "r", encoding="utf-8") as f:
            memories = json.load(f)
            for user_name, user_memory in memories.items():
                # if user_name != name:
                #     continue
                if 'history' not in user_memory.keys():
                    continue

                self.memory_bank[user_name] = copy.deepcopy(self.user_memory)
                if 'summary' in user_memory.keys():
                    self.memory_bank[user_name]['summary'] = user_memory['summary']
                if 'overall_history' in user_memory.keys():
                    self.memory_bank[user_name]['overall_history'] = user_memory['overall_history']
                if 'overall_personality' in user_memory.keys():
                    self.memory_bank[user_name]['overall_personality'] = user_memory['overall_personality']

                for date, content in user_memory['history'].items():
                    metadata = self._get_metadata(date)
                    memory_str = f'时间{date}的对话内容：' if self.language=='cn' else f'Conversation content on {date}:'
                    user_kw = '[|用户|]：' if self.language=='cn' else '[|User|]:'
                    ai_kw = '[|AI恋人|]：' if self.language=='cn' else '[|AI|]:'
                    for query, response in content:
                        tmp_str = memory_str
                        tmp_str += f'{user_kw} {query.strip()}; '
                        tmp_str += f'{ai_kw} {response.strip()}'
                        self.memory_bank[user_name]['history']['level_0'].append({'page_content':tmp_str,'metadata':metadata})
                    # memory_str += '\n'
                    if 'summary' in user_memory.keys():
                        if date in user_memory['summary'].keys():
                            summary = f'时间{date}的对话总结为：{user_memory["summary"][date]}' if self.language=='cn' else f'The summary of the conversation on {date} is: {user_memory["summary"][date]}'
                            self.memory_bank[user_name]['history']['level_0'].append({'page_content':summary,'metadata':metadata})
                        # Document(page_content=memory_str, metadata=metadata)) 
        self.write_memories(self.filepath.replace('.json' , '_forget_format.json'))

    def load_forget_file(self):
        forget_file = self.filepath.replace('.json' , '_forget_format.json')
        with open(forget_file, "r", encoding="utf-8") as f:
            self.memory_bank = json.load(f)
        return 
    
def get_docs_with_score(docs_with_score):
    docs=[]
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i-1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists

def load_memory_file(filepath,user_name,cur_date='',language='cn'):
    memory_loader = MemoryForgetterLoader(filepath,language)
    # docs = loader.load(user_name)
    textsplitter = ChineseTextSplitter(pdf=False)
    # if not os.path.exists(filepath.replace('.json','_forget_format.json')):
    print('write to ',os.path.exists(filepath.replace('.json','_forget_format.json')))
    
    # if not os.path.exists(os.path.exists(filepath.replace('.json','_forget_format.json'))):
    memory_loader.initial_load_and_save(name=user_name)
    # else:
    #     memory_loader.load_forget_file()

    docs = memory_loader.load_update_and_split(textsplitter,name=user_name,cur_date=cur_date)
    
    return docs, memory_loader 
    

def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, len(docs)-i)):
                for l in [i+k, i-k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        # print(doc0.metadata)
                        # exit()
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break
                        # print(doc0)
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
        id_list = sorted(list(id_set))
        id_lists = seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append((doc, scores[0][j]))
        return docs

class LocalMemoryRetrieval:
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL_CN,
                 embedding_device=EMBEDDING_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 language='cn'
                 ):
        self.language = language
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.user = ''
        self.top_k = top_k
        self.memory_loader = None
        self.memory_path = ''
        
    
    def init_memory_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    user_name: str = None,
                                    cur_date:str=None):
        loaded_files = []
        # filepath = filepath.replace('user',user_name)
        # vs_path = vs_path.replace('user',user_name)
        self.user=user_name
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None, None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                # try:
                self.memory_path = filepath
                docs,memory_loader = load_memory_file(filepath,user_name,cur_date,self.language)
                print(f"{file} 已成功加载")
                loaded_files.append(filepath)
                # except Exception as e:
                #     print(e)
                #     print(f"{file} 未能成功加载")
                #     return None
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    # if user_name not in fullfilepath:
                    #     continue
                    try:
                        self.memory_path = fullfilepath
                        new_docs, memory_loader = load_memory_file(fullfilepath,user_name,cur_date,self.language)
                        docs += new_docs
                        print(f"{file} 已成功加载")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} 未能成功加载")
        else:
            docs = []
            for file in filepath:
                try:
                    self.memory_path = file
                    new_docs,memory_loader = load_memory_file(file,user_name,cur_date,self.language)
                    docs += new_docs
                    print(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
        self.memory_loader = memory_loader
        if len(docs) > 0:
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                print(f'Load from previous memory index {vs_path}.')
                vector_store.add_documents(docs)
            else:
                if not vs_path:
                    vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
                vector_store = FAISS.from_documents(docs, self.embeddings)
            print(f'Saving to {vs_path}')
            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files
    
    def load_memory_index(self,vs_path):
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size=self.chunk_size
        return vector_store
    
    def search_memory(self,
                    query,
                    vector_store,
                    cur_date=''):

        # vector_store = FAISS.load_local(vs_path, self.embeddings)
        # FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        # vector_store.chunk_size=self.chunk_size
        related_docs_with_score = vector_store.similarity_search_with_score(query,
                                                                            k=self.top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        related_docs = sorted(related_docs, key=lambda x: x.metadata["source"], reverse=False)
        pre_date = ''
        date_docs = []
        dates = []
        cur_date = cur_date if cur_date else datetime.date.today().strftime("%Y-%m-%d") 
        for doc in related_docs:
            self.memory_loader.update_memory_when_searched(doc,user=self.user,cur_date=cur_date)
            #saving updated memory
            self.save_updated_memory()
            doc.page_content = doc.page_content.replace(f'时间{doc.metadata["source"]}的对话内容：','').strip()
            if doc.metadata["source"] != pre_date:
                # date_docs.append(f'在时间{doc.metadata["source"]}的回忆内容是：{doc.page_content}')
                date_docs.append(doc.page_content)
                pre_date = doc.metadata["source"]
                dates.append(pre_date)
            else:
                date_docs[-1] += f'\n{doc.page_content}' 
        # memory_contents = [doc.page_content for doc in related_docs]
        # memory_contents = [f'在时间'+doc.metadata['source']+'的回忆内容是：'+doc.page_content for doc in related_docs]
        return date_docs, ', '.join(dates) 
    
    def save_updated_memory(self):
        self.memory_loader.write_memories(self.memory_path.replace('.json' , '_forget_format.json'))
    