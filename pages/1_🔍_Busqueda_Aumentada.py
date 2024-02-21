from tiktoken import get_encoding, encoding_for_model
from utils.weaviate_interface_v3_spa import WeaviateClient, WhereFilter
from utils.prompt_templates_spa import question_answering_prompt_series_spa
from utils.openai_interface_spa import GPT_Turbo
from openai import BadRequestError
from utils.app_features_spa import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data, expand_content)
from utils.reranker_spa import ReRanker
from loguru import logger 
import streamlit as st
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)

## PAGE CONFIGURATION
st.set_page_config(page_title="Busqueda Aumentada",
                   page_icon="🔍",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)

## DATA + CACHE
data_path = 'data/1_IIEE_1_json_data_19_02_2024_22-17-49.json'
cache_path = ''
data = load_data(data_path)
cache = None  # Initialize cache as None

# Check if the cache file exists before attempting to load it
if os.path.exists(cache_path):
    cache = load_content_cache(cache_path)
else:
    logger.warning(f"Cache file {cache_path} not found. Proceeding without cache.")

#creates list of guests for sidebar
guest_list = sorted(list(set([d['document_title'] for d in data])))

with st.sidebar:
    st.subheader("Selecciona tu Base de datos 🗃️")
    client_type = st.radio(
        "Selecciona el modo de acceso:",
        ('Cloud', 'Local'),
        help='Elige un repositorio para determinar el conjunto de datos sobre el cual realizarás tu búsqueda. "Cloud" te permite acceder a datos alojados en nuestros servidores seguros, mientras que "Local" es para trabajar con datos alojados localmente en tu máquina.'
    )
if client_type == 'Cloud':
    api_key = st.secrets['WEAVIATE_CLOUD_API_KEY']
    url = st.secrets['WEAVIATE_CLOUD_ENDPOINT']

    weaviate_client = WeaviateClient(
        endpoint=url,
        api_key=api_key,
        # model_name_or_path='./models/finetuned-all-MiniLM-L6-v2-300',
        model_name_or_path="intfloat/multilingual-e5-small",
        # openai_api_key=os.environ['OPENAI_API_KEY']
        )
    available_classes=sorted(weaviate_client.show_classes())
    logger.info(available_classes)
    logger.info(f"Endpoint: {client_type} | Classes: {available_classes}")
elif client_type == 'Local':
    url = st.secrets['WEAVIATE_LOCAL_ENDPOINT']
    weaviate_client = WeaviateClient(
        endpoint=url,
        # api_key=api_key,
        # model_name_or_path='./models/finetuned-all-MiniLM-L6-v2-300',
        model_name_or_path="intfloat/multilingual-e5-small",
        # openai_api_key=os.environ['OPENAI_API_KEY']
        )
    available_classes=sorted(weaviate_client.show_classes())
    logger.info(f"Endpoint: {client_type} | Classes: {available_classes}")

def main():
    
    # Define the available user selected options
    available_models = ['gpt-3.5-turbo', 'gpt-4-1106-preview']
    # Define system prompts

    # Initialize selected options in session state
    if "openai_data_model" not in st.session_state:
        st.session_state["openai_data_model"] = available_models[0]
    
    if 'class_name' not in st.session_state:
        st.session_state['class_name'] = None

    with st.sidebar:
        st.session_state['class_name'] = st.selectbox(
            label='Repositorio:',
            options=available_classes,
            index=None,
            placeholder='Repositorio',
            help='Elige un repositorio para determinar el conjunto de datos sobre el cual realizarás tu búsqueda. "Cloud" te permite acceder a datos alojados en nuestros servidores seguros, mientras que "Local" es para trabajar con datos alojados localmente en tu máquina.'
            )
        # Check if the collection name has been selected
        class_name = st.session_state['class_name']
        if class_name:
            st.success(f"Repositorio seleccionado ✅: {st.session_state['class_name']}")

        else:
            st.warning("🎗️ No olvides seleccionar el repositorio 👆 a consultar 🗄️.")
            st.stop()  # Stop execution of the script
        
        model_choice = st.selectbox(
            label="Elige un modelo de OpenAI",
            options=available_models,
            index= available_models.index(st.session_state["openai_data_model"]),
            help='Escoge entre diferentes modelos de OpenAI para generar respuestas a tus consultas. Cada modelo tiene distintas capacidades y limitaciones.'
        )
        st.sidebar.make_llm_call = st.checkbox(
        label="Activar GPT",
        help='Marca esta casilla para activar la generación de texto con GPT. Esto te permitirá obtener respuestas automáticas a tus consultas.'
        )

        with st.expander("Filtros de Busqueda"):
            guest_input = st.selectbox(
                label='Selección de documentos',
                options=guest_list,
                index=None,
                placeholder='Documento',
                help='Elige un documento específico del repositorio para afinar tu búsqueda a datos relevantes.'
            )

        with st.expander("Parametros de Busqueda"):
            retriever_choice = st.selectbox(
            label="Selecciona un método",
            options=["Hybrid", "Vector", "Keyword"],
            help='Determina el método de recuperación de información: "Hybrid" combina búsqueda por palabras clave y por similitud semántica, "Vector" usa embeddings de texto para encontrar coincidencias semánticas, y "Keyword" realiza una búsqueda tradicional por palabras clave.'
            )
            
            reranker_enabled = st.checkbox(
                label="Activar Reranker",
                value=True,
                help='Activa esta opción para ordenar los resultados de la búsqueda según su relevancia, utilizando un modelo de reordenamiento adicional.'
            )

            alpha_input = st.slider(
                label='Alpha para motor hibrido',
                min_value=0.00,
                max_value=1.00,
                value=0.40,
                step=0.05,
                help='Ajusta el parámetro alfa para equilibrar los resultados entre los métodos de búsqueda por vector y por palabra clave en el motor híbrido.'
            )
            
            retrieval_limit = st.slider(
                label='Resultados a Reranker',
                min_value=10,
                max_value=300,
                value=100,
                step=10,
                help='Establece el número de resultados que se recuperarán antes de aplicar el reordenamiento.'
            )
            
            top_k_limit = st.slider(
                label='Top K Limit',
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help='Define el número máximo de resultados a mostrar después de aplicar el reordenamiento.'
            )
            
            temperature_input = st.slider(
                label='Temperatura',
                min_value=0.0,
                max_value=1.0,
                value=0.10,
                step=0.10,
                help='Ajusta la temperatura para la generación de texto con GPT, lo que influirá en la creatividad de las respuestas.'
            )

    logger.info(weaviate_client.display_properties)

    def perform_search(client, retriever_choice, query, class_name, search_limit, guest_filter, display_properties, alpha_input):
        if retriever_choice == "Keyword":
            return weaviate_client.keyword_search(
                request=query,
                class_name=class_name,
                limit=search_limit,
                where_filter=guest_filter,
                display_properties=display_properties
            ), "Resultados de la Busqueda - Motor: Keyword: "
        elif retriever_choice == "Vector":
            return weaviate_client.vector_search(
                request=query,
                class_name=class_name,
                limit=search_limit,
                where_filter=guest_filter,
                display_properties=display_properties
            ), "Resultados de la Busqueda - Motor: Vector"
        elif retriever_choice == "Hybrid":
            return weaviate_client.hybrid_search(
                request=query,
                class_name=class_name,
                alpha=alpha_input,
                limit=search_limit,
                properties=["content"],
                where_filter=guest_filter,
                display_properties=display_properties
            ), "Resultados de la Busqueda - Motor: Hybrid"


    ## RERANKER
    reranker = ReRanker(model_name='BAAI/bge-reranker-base')

    ## LLM
    model_name = model_choice
    llm = GPT_Turbo(model=model_name, api_key=st.secrets['OPENAI_API_KEY'])
    encoding = encoding_for_model(model_name)


    ########################
    ## SETUP MAIN DISPLAY ##
    ########################
    st.image('./static/images/cervezas-mahou.jpeg', width=300)
    st.subheader(f"✨🔍📚 **Búsqueda Aumentada** 📖🔍✨ Impuestos Especiales ")
    st.caption("Descubre insights ocultos y responde a tus preguntas especializadas utilizando el poder de la IA")
    st.write('\n')
    
    query = st.text_input('Escribe tu pregunta aquí: ')
    st.write('\n\n\n\n\n')

    ############
    ## SEARCH ##
    ############
    if query:
        # make hybrid call to weaviate
        guest_filter = WhereFilter(
            path=['document_title'],
            operator='Equal',
            valueText=guest_input).todict() if guest_input else None
        
        
        # Determine the appropriate limit based on reranking
        search_limit = retrieval_limit if reranker_enabled else top_k_limit

        # Perform the search
        query_response, subheader_msg = perform_search(
            client=weaviate_client,
            retriever_choice=retriever_choice,
            query=query,
            class_name=class_name,
            search_limit=search_limit,
            guest_filter=guest_filter,
            display_properties=weaviate_client.display_properties,
            alpha_input=alpha_input if retriever_choice == "Hybrid" else None
            )
        
        
        # Rerank the results if enabled
        if reranker_enabled:
            search_results = reranker.rerank(
                results=query_response,
                query=query,
                apply_sigmoid=True,
                top_k=top_k_limit
            )
            subheader_msg += " Reranked"
        else:
            # Use the results directly if reranking is not enabled
            search_results = query_response

        logger.info(search_results)
        expanded_response = expand_content(search_results, cache, content_key='doc_id', create_new_list=True)

        # validate token count is below threshold
        token_threshold = 8000 if model_name == 'gpt-3.5-turbo-16k' else 3500
        valid_response = validate_token_threshold(
            ranked_results=expanded_response,
            base_prompt=question_answering_prompt_series_spa,
            query=query,
            tokenizer=encoding,
            token_threshold=token_threshold,
            verbose=True
        )
        logger.info(valid_response)
        #########
        ## LLM ##
        #########
        make_llm_call = st.sidebar.make_llm_call
        # prep for streaming response
        st.subheader("Respuesta GPT:")
        with st.spinner('Generando Respuesta...'):
            st.markdown("----")
            # Creates container for LLM response
            chat_container, response_box = [], st.empty()

            # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response)
            # logger.info(prompt)
            if make_llm_call:

                try:
                    for resp in llm.get_chat_completion(
                        prompt=prompt,
                        temperature=temperature_input,
                        max_tokens=350, # expand for more verbose answers
                        show_response=True,
                        stream=True):

                        # inserts chat stream from LLM
                        with response_box:
                            content = resp.choices[0].delta.content
                            if content:
                                chat_container.append(content)
                                result = "".join(chat_container).strip()
                                st.write(f'{result}')
                except BadRequestError:
                    logger.info('Making request with smaller context...')
                    valid_response = validate_token_threshold(
                        ranked_results=search_results,
                        base_prompt=question_answering_prompt_series_spa,
                        query=query,
                        tokenizer=encoding,
                        token_threshold=token_threshold,
                        verbose=True
                    )

                    # generate LLM prompt
                    prompt = generate_prompt_series(query=query, results=valid_response)
                    for resp in llm.get_chat_completion(
                        prompt=prompt,
                        temperature=temperature_input,
                        max_tokens=350, # expand for more verbose answers
                        show_response=True,
                        stream=True):

                        try:
                            # inserts chat stream from LLM
                            with response_box:
                                content = resp.choices[0].delta.content
                                if content:
                                    chat_container.append(content)
                                    result = "".join(chat_container).strip()
                                    st.write(f'{result}')
                        except Exception as e:
                            print(e)

            ####################
            ## Search Results ##
            ####################
            st.subheader(subheader_msg)
            for i, hit in enumerate(search_results):
                col1, col2 = st.columns([7, 3], gap='large')
                page_url = hit['page_url']
                page_label = hit['page_label']
                document_title = hit['document_title']
                # Assuming 'page_summary' is available and you want to display it
                page_summary = hit.get('page_summary', 'Summary not available')

                with col1:
                    st.markdown(f'''
                                <span style="color: #3498db; font-size: 19px; font-weight: bold;">{document_title}</span><br>
                                {page_summary}
                                [**Página:** {page_label}]({page_url})
                            ''', unsafe_allow_html=True)

                    with st.expander("📄 Clic aquí para ver contexto:"):
                        try:
                            content = hit['content']
                            st.write(content)
                        except Exception as e:
                            st.write(f"Error displaying content: {e}")

                # with col2:
                #     # If you have an image or want to display a placeholder image
                #     image = "URL_TO_A_PLACEHOLDER_IMAGE"  # Replace with a relevant image URL if applicable
                #     st.image(image, caption=document_title, width=200, use_column_width=False)
                #     st.markdown(f'''
                #                 <p style="text-align: right;">
                #                     <b>Document Title:</b> {document_title}<br>
                #                     <b>File Name:</b> {file_name}<br>
                #                 </p>''', unsafe_allow_html=True)



if __name__ == '__main__':
    main()