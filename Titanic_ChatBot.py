import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import streamlit as st
import io

# Load Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# FastAPI backend
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

# Initialize LangChain with OpenAI (replace with your API key)
chatbot = ChatOpenAI(openai_api_key="your-api-key")

@app.post("/query")
def query_titanic(data: QueryRequest):
    question = data.question.lower()
    response = "Sorry, I can't answer that."
    
    if "percentage of passengers were male" in question:
        male_pct = (df['Sex'] == 'male').mean() * 100
        response = f"{male_pct:.2f}% of the passengers were male."
    
    elif "histogram of passenger ages" in question:
        fig, ax = plt.subplots()
        sns.histplot(df['Age'].dropna(), bins=20, ax=ax)
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        return img_buf.getvalue()
    
    elif "average ticket fare" in question:
        avg_fare = df['Fare'].mean()
        response = f"The average ticket fare was ${avg_fare:.2f}."
    
    elif "passengers embarked from each port" in question:
        embarked_counts = df['Embarked'].value_counts().to_dict()
        response = f"Passenger embarkations: {embarked_counts}"
    
    return {"answer": response}

# Streamlit frontend
def main():
    st.title("Titanic Chatbot")
    user_input = st.text_input("Ask me about the Titanic dataset")
    
    if user_input:
        response = query_titanic(QueryRequest(question=user_input))
        if isinstance(response, dict):
            st.write(response["answer"])
        else:
            st.image(response, caption="Histogram of Passenger Ages")

if __name__ == "__main__":
    main()
