import gradio as gr
import tensorflow as tf
import pickle

from attention import AttentionLayer
from preprocess import preprocess_text

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load models
models = {
    "Simple RNN": tf.keras.models.load_model("models/simple_rnn.keras"),
    "LSTM": tf.keras.models.load_model("models/lstm_model.keras"),
    "GRU + Attention": tf.keras.models.load_model(
        "models/gru_attention.keras",
        custom_objects={"AttentionLayer": AttentionLayer}
    )
}

def predict(text, selected_model):
    processed = preprocess_text(text, tokenizer)

    def single_result(name, prob):
        sentiment = "Positive 😊" if prob > 0.5 else "Negative 😞"
        return f"""
###  {name}
**Sentiment:** {sentiment}  
**Confidence:** `{prob:.4f}`
"""

    if selected_model == "All Models":
        results = []
        for name, model in models.items():
            prob = model.predict(processed, verbose=0)[0][0]
            results.append(single_result(name, prob))
        return "\n---\n".join(results)

    prob = models[selected_model].predict(processed, verbose=0)[0][0]
    return single_result(selected_model, prob)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎬 Multi-Model Sentiment Analysis
        Compare **Simple RNN**, **LSTM**, and **GRU + Attention** models trained on IMDB reviews.

        --Enter a movie review and choose a model  
        -- Select **All Models** to compare predictions
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=5,
                placeholder="Type a movie review here...",
                label="Movie Review"
            )
            model_choice = gr.Dropdown(
                ["Simple RNN", "LSTM", "GRU + Attention", "All Models"],
                value="GRU + Attention",
                label="Select Model"
            )
            submit_btn = gr.Button("Analyze Sentiment")

        with gr.Column(scale=3):
            output = gr.Markdown(label="Prediction Result")

    submit_btn.click(
        fn=predict,
        inputs=[text_input, model_choice],
        outputs=output
    )

demo.launch()
