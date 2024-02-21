from tiktoken import get_encoding, encoding_for_model
from utils.weaviate_interface_v3_spa import WeaviateClient, WhereFilter
from utils.prompt_templates_spa import question_answering_prompt_series_spa
from utils.openai_interface_spa import GPT_Turbo
from openai import BadRequestError
from utils.app_features_spa import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data, expand_content)
from utils.reranker_spa import ReRanker
from openai import OpenAI

from loguru import logger 
import streamlit as st
import os
import utils.system_prompts as system_prompts
import base64
import json

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)

## PAGE CONFIGURATION
st.set_page_config(page_title="Busqueda Conversacional",
                   page_icon="🗣",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)

def encode_image(uploaded_file):
  return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

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

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def main():
    
    # Define the available user selected options
    available_models = ['gpt-3.5-turbo', 'gpt-4-1106-preview']
    # Define system prompts
    system_prompt_list = ["🤖ChatGPT","🧙🏾‍♂️Professor Synapse", "👩🏼‍💼Marketing Jane"]


    # Initialize selected options in session state
    if "openai_data_model" not in st.session_state:
        st.session_state["openai_data_model"] = available_models[0]
    if "system_prompt_data_list" not in st.session_state and "system_prompt_data_model" not in st.session_state:
        # This should be the emoji string the user selected
        st.session_state["system_prompt_data_list"] = system_prompt_list[0]
        # Now we get the corresponding prompt variable using the selected emoji string
        st.session_state["system_prompt_data_model"] = system_prompts.prompt_mapping[system_prompt_list[0]]

    # logger.debug(f"Assistant: {st.session_state['system_prompt_sync_list']}")
    # logger.debug(f"System Prompt: {st.session_state['system_prompt_sync_model']}")

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
        
        system_prompt = st.selectbox(
                label="Elige un asistente",
                options=system_prompt_list,
                index=system_prompt_list.index(st.session_state["system_prompt_data_list"]),
        )
        
        with st.expander("Filtros de Busqueda"):
            guest_input = st.selectbox(
                label='Selección de Documento',
                options=guest_list,
                index=None,
                placeholder='Documentos',
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
                value=0.20,
                step=0.10,
                help='Ajusta la temperatura para la generación de texto con GPT, lo que influirá en la creatividad de las respuestas.'
            )

    # Update the model choice in session state
    if st.session_state["openai_data_model"]!=model_choice:
        st.session_state["openai_data_model"] = model_choice
    logger.info(f"Data model: {st.session_state['openai_data_model']}")

    # Update the system prompt choice in session state
    if st.session_state["system_prompt_data_list"] != system_prompt:
        # This should be the emoji string the user selected
        st.session_state["system_prompt_data_list"] = system_prompt  
        # Now we get the corresponding prompt variable using the selected emoji string
        selected_prompt_variable = system_prompts.prompt_mapping[system_prompt]
        st.session_state['system_prompt_data_model'] = selected_prompt_variable
        # logger.info(f"System Prompt: {selected_prompt_variable}")
    logger.info(f"Assistant: {st.session_state['system_prompt_data_list']}")
    # logger.info(f"System Prompt: {st.session_state['system_prompt_sync_model']}")

    logger.info(weaviate_client.display_properties)

    def database_search(query):
        # Determine the appropriate limit based on reranking
        search_limit = retrieval_limit if reranker_enabled else top_k_limit
        
        # make hybrid call to weaviate
        guest_filter = WhereFilter(
            path=['document_title'],
            operator='Equal',
            valueText=guest_input).todict() if guest_input else None

        try:
            # Perform the search based on retriever_choice
            if retriever_choice == "Keyword":
                query_results = weaviate_client.keyword_search(
                    request=query,
                    class_name=class_name,
                    limit=search_limit,
                    where_filter=guest_filter
                )
            elif retriever_choice == "Vector":
                query_results = weaviate_client.vector_search(
                    request=query,
                    class_name=class_name,
                    limit=search_limit,
                    where_filter=guest_filter
                )
            elif retriever_choice == "Hybrid":
                query_results = weaviate_client.hybrid_search(
                    request=query,
                    class_name=class_name,
                    alpha=alpha_input,
                    limit=search_limit,
                    properties=["content"],
                    where_filter=guest_filter
                )
            else:
                return json.dumps({"error": "Invalid retriever choice"})


            ## RERANKER
            reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-12-v2')
            model_name = model_choice
            encoding = encoding_for_model(model_name)
            
            # Rerank the results if enabled
            if reranker_enabled:
                search_results = reranker.rerank(
                    results=query_results,
                    query=query,
                    apply_sigmoid=True,
                    top_k=top_k_limit
                )
            
            else:
                # Use the results directly if reranking is not enabled
                search_results = query_results

            # logger.debug(search_results)
            # Save search results to session state for later use
            # st.session_state['search_results'] = search_results
            add_to_search_history(query=query, search_results=search_results)
            expanded_response = expand_content(search_results, cache, content_key='doc_id', create_new_list=True)

            # validate token count is below threshold
            token_threshold = 8000
            valid_response = validate_token_threshold(
                ranked_results=expanded_response,
                base_prompt=question_answering_prompt_series_spa,
                query=query,
                tokenizer=encoding,
                token_threshold=token_threshold,
                verbose=True
            )
            
            # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response)

            # If the strings in 'prompt' are double-escaped, decode them before dumping to JSON
            # prompt_decoded = prompt.encode().decode('unicode_escape')
            
            # Then, when you dump to JSON, it should no longer double-escape the characters
            return json.dumps({
                    "query": query,
                    "Search Results": prompt,
                }, ensure_ascii=False)
    
        except Exception as e:
            # Handle any exceptions and return a JSON formatted error message
            return json.dumps({
                "error": "An error occurred during the search",
                "details": str(e)
            })

    # When a new message is added, include the type and content
    def add_to_search_history(query, search_results):
        st.session_state["data_search_history"].append({
            "query": query,
            "search_results": search_results,
        })
    
    # Function to display search results
    def display_search_results():
        # Loop through each item in the search history
        for search in st.session_state['data_search_history']:
            query = search["query"]
            search_results = search["search_results"]
            # Create an expander for each search query
            with st.expander(f"Pregunta: {query}", expanded=False):
                for i, hit in enumerate(search_results):
                    # col1, col2 = st.columns([7, 3], gap='large')
                    page_url = hit['page_url']
                    page_label = hit['page_label']
                    document_title = hit['document_title']
                    # Assuming 'page_summary' is available and you want to display it
                    page_summary = hit.get('page_summary', 'Summary not available')

                    # with col1:
                    st.markdown(f'''
                            <span style="color: #3498db; font-size: 19px; font-weight: bold;">{document_title}</span><br>
                            {page_summary}
                            [**Página:** {page_label}]({page_url})
                        ''', unsafe_allow_html=True)

                        # with st.expander("📄 Clic aquí para ver contexto:"):
                        #     try:
                        #         content = hit['content']
                        #         st.write(content)
                        #     except Exception as e:
                        #         st.write(f"Error displaying content: {e}")

                    # with col2:
                    #     # If you have an image or want to display a placeholder image
                    #     image = "URL_TO_A_PLACEHOLDER_IMAGE"  # Replace with a relevant image URL if applicable
                    #     st.image(image, caption=document_title, width=200, use_column_width=False)
                    #     st.markdown(f'''
                    #                 <p style="text-align: right;">
                    #                     <b>Document Title:</b> {document_title}<br>
                    #                     <b>File Name:</b> {file_name}<br>
                    #                 </p>''', unsafe_allow_html=True)
    
    ########################
    ## SETUP MAIN DISPLAY ##
    ########################
    
    st.image('./static/images/cervezas-mahou.jpeg', width=400)
    st.subheader(f"✨🗣️📘 **Búsqueda Conversacional** 💡🗣️✨ - Impuestos Especiales")
    st.write('\n')
    col1, col2 = st.columns([50,50])

    # Initialize chat history
    if "data_chat_history" not in st.session_state:
        st.session_state["data_chat_history"] = []
    
    if "data_search_history" not in st.session_state:
        st.session_state["data_search_history"] = []

    with col1:
        st.write("Chat History:")
        # Create a container for chat history
        chat_history_container = st.container(height=500, border=True)
        # Display chat messages from history on app rerun
        with chat_history_container:
            for message in st.session_state["data_chat_history"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    # Function to update chat display
    def update_chat_display():
        with chat_history_container:
            for message in st.session_state["data_chat_history"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
        st.session_state["data_chat_history"].append({"role": "user", "content": prompt})
        # Initially display the chat history
        update_chat_display()
        # # Display user message in chat message container
        # with st.chat_message("user"):
        #     st.markdown(prompt)

        with st.spinner('Generando Respuesta...'):
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "database_search",
                        "description": "Takes the users query about the database and returns the results, extracting info to answer the user's question",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                    "query": {"type": "string", "description": "query"},  
                                    
                                },
                            "required": ["query"],
                        },
                    }
                }
            ]

            # Display live assistant response in chat message container
            with st.chat_message(
                name="assistant",
                avatar="./static/images/openai_purple_logo_hres.jpeg"):
                message_placeholder = st.empty()

            # Building the messages payload with proper OPENAI API structure
            messages=[
                    {"role": "system", "content": st.session_state["system_prompt_data_model"]}
                ] + [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state["data_chat_history"]
                ]
            logger.debug(f"Initial Messages: {messages}")
            # call the OpenAI API to get the response
            
            RESPONSE = client.chat.completions.create(
                model=st.session_state["openai_data_model"],
                temperature=0.5,
                messages=messages,
                tools=tools,
                tool_choice="auto",  # auto is default, but we'll be explicit
                stream=True
            )
            logger.debug(f"First Response: {RESPONSE}") 


            FULL_RESPONSE = ""
            tool_calls = []
            # build up the response structs from the streamed response, simultaneously sending message chunks to the browser
            for chunk in RESPONSE:
                delta = chunk.choices[0].delta
                # logger.debug(f"chunk: {delta}")

                

                if delta and delta.content:
                    text_chunk = delta.content
                    FULL_RESPONSE += str(text_chunk)
                    message_placeholder.markdown(FULL_RESPONSE + "▌")
                
                elif delta and delta.tool_calls:
                    tcchunklist = delta.tool_calls
                    for tcchunk in tcchunklist:
                        if len(tool_calls) <= tcchunk.index:
                            tool_calls.append({"id": "", "type": "function", "function": { "name": "", "arguments": "" } })
                        tc = tool_calls[tcchunk.index]

                        if tcchunk.id:
                            tc["id"] += tcchunk.id
                        if tcchunk.function.name:
                            tc["function"]["name"] += tcchunk.function.name
                        if tcchunk.function.arguments:
                            tc["function"]["arguments"] += tcchunk.function.arguments
            if tool_calls:
                logger.debug(f"tool_calls: {tool_calls}")
                # Define a dictionary mapping function names to actual functions
                available_functions = {
                    "database_search": database_search,
                    # Add other functions as necessary
                }
                available_functions = {
                    "database_search": database_search,
                }  # only one function in this example, but you can have multiple
                logger.debug(f"FuncCall Before messages: {messages}")
                # Process each tool call
                for tool_call in tool_calls:
                    # Get the function name and arguments from the tool call
                    function_name = tool_call['function']['name']
                    function_args = json.loads(tool_call['function']['arguments'])
                    
                    # Get the actual function to call
                    function_to_call = available_functions[function_name]

                    # Call the function and get the response
                    function_response = function_to_call(**function_args)

                    # Append the function response to the messages list
                    messages.append({
                        "role": "assistant",
                        "tool_call_id": tool_call['id'],
                        "name": function_name,
                        "content": function_response,
                    })
                logger.debug(f"FuncCall After messages: {messages}")

                RESPONSE = client.chat.completions.create(
                    model=st.session_state["openai_data_model"],
                    temperature=0.1,
                    messages=messages,
                    stream=True
                )
                logger.debug(f"Second Response: {RESPONSE}") 

                # build up the response structs from the streamed response, simultaneously sending message chunks to the browser
                for chunk in RESPONSE:
                    delta = chunk.choices[0].delta
                    # logger.debug(f"chunk: {delta}")

                    if delta and delta.content:
                        text_chunk = delta.content
                        FULL_RESPONSE += str(text_chunk)
                        message_placeholder.markdown(FULL_RESPONSE + "▌")
        # Add assistant response to chat history
        st.session_state["data_chat_history"].append({"role": "assistant", "content": FULL_RESPONSE})
        logger.debug(f"chat_history: {st.session_state['data_chat_history']}")
    
# Next block of code...

        
    ####################
    ## Search Results ##
    ####################
    # st.subheader(subheader_msg)
    with col2:
        st.write("Search Results:")
        with st.container(height=500, border=True):
            # Check if 'data_search_history' is in the session state and not empty
                if 'data_search_history' in st.session_state and st.session_state['data_search_history']:
                    display_search_results()
                    # # Extract the latest message from the search history
                    #     latest_search = st.session_state['data_search_history'][-1]
                    #     query = latest_search["query"]
                    #     with st.expander(query, expanded=False):
                    #         # Extract the latest message from the search history
                    #         latest_search = st.session_state['data_search_history'][-1]
                    #         query = latest_search["query"]
                    #         for i, hit in enumerate(latest_search["search_results"]):
                    #             col1, col2 = st.columns([7, 3], gap='large')
                    #             episode_url = hit['episode_url']
                    #             title = hit['title']
                    #             guest=hit['guest']
                    #             show_length = hit['length']
                    #             time_string = convert_seconds(show_length)
                    #             # content = ranked_response[i]['content'] # Get 'content' from the same index in ranked_response
                    #             content = hit['content']
                            
                    #             with col1:
                    #                 st.write( search_result(i=i, 
                    #                                         url=episode_url,
                    #                                         guest=guest,
                    #                                         title=title,
                    #                                         content=content,
                    #                                         length=time_string),
                    #                                         unsafe_allow_html=True)
                    #                 st.write('\n\n')

                    #                 # with st.container("Episode Summary:"):
                    #                 #     try:
                    #                 #         ep_summary = hit['summary']
                    #                 #         st.write(ep_summary)
                    #                 #     except Exception as e:
                    #                 #         st.error(f"Error displaying summary: {e}")

                    #             with col2:
                    #                 image = hit['thumbnail_url']
                    #                 st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)
                    #                 st.markdown(f'''
                    #                             <p style="text-align: right;">
                    #                                 <b>Episode:</b> {title.split('|')[0]}<br>
                    #                                 <b>Guest:</b> {hit['guest']}<br>
                    #                                 <b>Length:</b> {time_string}
                    #                             </p>''', unsafe_allow_html=True)


if __name__ == '__main__':
    main()