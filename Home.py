import streamlit as st
import base64

## PAGE CONFIGURATION
st.set_page_config(page_title="Búsqueda Aumentada MSM para Impuestos Especiales",
                   page_icon="🔍",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

st.image('./static/images/cervezas-mahou.jpeg', width=700,)

# Mensaje de bienvenida
st.markdown(
    """
    # ¡Bienvenido a Búsqueda Aumentada MSM para Impuestos Especiales! 🔍📚
    
    Esta aplicación es una herramienta diseñada específicamente para la exploración y análisis de datos en el ámbito de Impuestos Especiales utilizando el poder de la Inteligencia Artificial.

    **👈 Selecciona una opción en la barra lateral** para comenzar a explorar las diferentes funcionalidades que ofrece la aplicación.
    """)
file_ = open('./static/images/screen_recording_busqueda_final_2.gif', "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.subheader("Uso de la Aplicación: 🔍 Busqueda Aumentada")
st.caption("Observa en acción cómo la busqueda aumentada con una potente IA simplifica la búsqueda de información, todo con una interfaz de usuario facíl de usar.")
st.markdown(
    f'<div style="text-align: center;"><img src="data:image/gif;base64,{data_url}" alt="demo gif" style="max-width: 100%; height: auto;"></div>',
    unsafe_allow_html=True,
)

st.markdown("""

    ### ¿Quieres aprender más?
    - Visita nuestra [página web](https://tupagina.com)
    - Sumérgete en nuestra [documentación](https://tudocumentacion.com)
    - Participa y pregunta en nuestros [foros comunitarios](https://tucomunidad.com)
    
    ### Explora demos más complejos
    - Descubre cómo aplicamos la IA para [analizar datasets especializados](https://tulinkdedataset.com)
    - Explora [bases de datos de acceso público](https://tulinkdedatasetpublico.com) y ve la IA en acción
    """,
    unsafe_allow_html=True
)
