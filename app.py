import os
from dotenv import load_dotenv
from getpass import getpass
from google import genai

# Load trained ML model and vectorizer 
import joblib
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

load_dotenv()

# Load api_key
def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    print("API key not found. Enter manually:")
    return getpass("Enter Gemini API key: ")

client = genai.Client(api_key=get_api_key())

# genAi analysis
def ask_llm(text):
    text = text[:500]  # limit tokens 
    response = client.models.generate_content(
        model="gemini-2.5-flash",   
        contents=f"""
        REAL or FAKE?
        Give only:
        Label: REAL or FAKE
        Reason: two short sentence

        {text}
        """
    )
    return response.text.strip()

# Taking input from user
def get_input() :
  input_ready = False
  while not input_ready:
    input_field = input("Enter news article (50–500 words): ")
    if not input_field :
      print("invalid")
      continue
    elif not (50<= len(input_field.split()) <= 500):
       print("DOES NOT MATCH INPUT REQUIREMENTS/MIN. WORDS 50 , MAX. WORDS 500")
       continue
    else:
      input_ready = True
  return input_field

# Processing input
def process_input(input_string):
  text = input_string
  text = text.lower()
  text_vec = vectorizer.transform([text])
  return text_vec

# Predicting 
def predict(text, text_vec):

    pred = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0]
    confidence = max(prob)

    label = "Fake News" if pred == 0 else "Real News"

    print("\nML Prediction:", label)
    print("Confidence:", round(confidence, 3))

    if confidence >= 0.80:
        print("Final Decision: ML")
    else:
        print("Final Decision: LLM (fallback)\n")

        llm_result = ask_llm(text)
        print("LLM Analysis:\n", llm_result)

if __name__ == "__main__":
    print("📰 Hybrid Fake News Detector (ML + LLM)")
    user_article_text = get_input()
    processed_article_vec = process_input(user_article_text)
    predict(user_article_text, processed_article_vec)
