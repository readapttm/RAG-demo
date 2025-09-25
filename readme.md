# Simple implementation of a RAG application 

This repo contains a minimal langchain application to demonstrate RAG using the annual letters to shareholders issued by Berkshire Hathaway as context.

The app can be launched by running main.ipynb. Enter your OpenAI API key when prompted by getpass.
You can choose how many of the downloaded pdf files to include by setting the document_limit parameter. You can also limit the total context tokens you supply to the LLM using the token_limit parameter.

Some questions you may want to try asking:

question = "What is the key to success in investing?"
question = "Can you give me some examples based on the context?"
question = "What are the highlights of 2022?" (assuming you have downloaded the relevant letter)

To avoid any potential issues I will not host the pdfs in this repo, but they are available at the website below. Simply create a folder named 'pdf_data' and fill it with as many of the letters as desired:
 https://www.berkshirehathaway.com/letters/letters.html.

 Alternatively, feel free to download a different set of pdfs and ask a different set of questions more appropriate for your chosen subject matter.

