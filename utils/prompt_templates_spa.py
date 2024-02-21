question_answering_prompt_series_spa = '''
Su tarea es sintetizar y razonar sobre una serie de contenidos proporcionados. 
Después de su síntesis, utilice estos contenidos para responder a la pregunta a continuación. La serie estará en el siguiente formato:\n
```
RESUMEN: <summary>
DOCUMENTO: <document>
CONTENIDO: <transcript>
```\n\n
Inicio de la Serie:
```
{series}
```
Pregunta:\n
{question}\n
Responda a la pregunta y proporcione razonamientos si es necesario para explicar la respuesta.
Si el contexto no proporciona suficiente información para responder a la pregunta, entonces
indique que no puede responder a la pregunta con el contexto proporcionado.

Respuesta:
'''

context_block_spa = '''
RESUMEN: {summary}
DOCUMENTO: {document}
CONTENIDO: {transcript}
'''
