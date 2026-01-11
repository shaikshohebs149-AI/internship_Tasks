from transformers import pipeline, set_seed

def generate_coherent_paragraph(topic_prompt, max_len=150):
    """
    Generates a coherent paragraph using the GPT-2 model.
    """
    # 1. Initialize the text-generation pipeline
    # This automatically loads the GPT-2 model and tokenizer
    generator = pipeline('text-generation', model='gpt2')
    
    # 2. Set a seed for reproducibility
    set_seed(42)
    
    # 3. Generate text
    # We use 'sampling' and 'top_k' to make the text more creative and less repetitive
    output = generator(
        topic_prompt, 
        max_length=max_len, 
        num_return_sequences=1,
        do_sample=True, 
        top_k=50, 
        top_p=0.95,
        temperature=0.8,
        truncation=True
    )
    
    return output[0]['generated_text']

# --- TEST THE MODEL ---
if __name__ == "__main__":
    user_topic = "The impact of renewable energy on global economies is"
    
    print(f"--- USER PROMPT --- \n{user_topic}")
    print("\n--- GENERATING... ---")
    
    paragraph = generate_coherent_paragraph(user_topic)
    
    print("\n--- GENERATED PARAGRAPH ---")
    print(paragraph)