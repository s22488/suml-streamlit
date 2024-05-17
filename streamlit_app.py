import pandas as pd
import streamlit as st
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

st.title("Praca domowa 04")

st.image("t5-image.png")

st.header("Przetwarzanie języka naturalnego")

st.write("Aplikacja służy do analizy i/lub tłumaczenia tekstu")

st.write("Za pomocą tej aplikacji możesz analizować wydźwięk emocjonalny tekstu lub przetłumaczyć go na język niemiecki.")

st.write("Podany tekst musi być w języku angielskim.")

st.subheader("Opcje:")
st.text("1. Wydźwięk emocjonalny tekstu (eng) - wybierz aby przeanalizować wydźwięk tekstu")
st.text("2. Tłumacz na język niemiecki (eng) - wybierz aby przetłumaczyć tekst")

st.info("Proces analizy lub tłumaczenia może chwilę potrwać.")

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumacz na język niemiecki (eng)",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        try:
            with st.spinner("Analiza..."):
                classifier = pipeline("sentiment-analysis")
                answer = classifier(text)

                st.success("Analiza zakończona!")
                st.write(answer)
        except Exception as e:
            st.error(f"Wystąpił błąd podczas analizy wydźwięku: {e}")

if option == "Tłumacz na język niemiecki (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        try:
            with st.spinner("Tłumaczenie..."):
                tokenizer = T5Tokenizer.from_pretrained("t5-base")
                model = T5ForConditionalGeneration.from_pretrained("t5-base")

                input_text = f"translate from English to German: {text}"
                input_ids = tokenizer.encode(input_text, return_tensors="pt")

                output_ids = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
                translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                st.success("Tłumaczenie zakończone!")
                st.write(f"Tłumaczenie: {translation}")
        except Exception as e:
            st.error(f"Wystąpił błąd podczas tłumaczenia tekstu: {e}")

st.write("Numer indeksu: 22488")
