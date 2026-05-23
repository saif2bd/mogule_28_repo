import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os

# =====================
# Load trained model
# =====================
model_path = "student_rf_pipeline.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please run rf_trian.py first.")

with open(model_path, "rb") as f:
    rf_pipeline = pickle.load(f)

print("✅ Model loaded successfully!")

# =====================
# Prediction function
# =====================
def predict_hsc_result(gender, age, address, famsize, pstatus, m_edu, f_edu, 
                       m_job, f_job, relationship, smoker, tuition_fee, 
                       time_friends, ssc_result):
    """
    Predict HSC result based on student features
    """
    try:
        # Create input dataframe with the same structure as training data
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [int(age)],
            'address': [address],
            'famsize': [famsize],
            'Pstatus': [pstatus],
            'M_Edu': [int(m_edu)],
            'F_Edu': [int(f_edu)],
            'M_Job': [m_job],
            'F_Job': [f_job],
            'relationship': [relationship],
            'smoker': [smoker],
            'tuition_fee': [int(tuition_fee)],
            'time_friends': [int(time_friends)],
            'ssc_result': [float(ssc_result)]
        })
        
        # Make prediction
        prediction = rf_pipeline.predict(input_data)[0]
        
        return f"**Predicted HSC Result: {prediction:.2f}**"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# =====================
# Gradio Interface
# =====================
with gr.Blocks(title="Bangladesh Student Performance Predictor", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🎓 Bangladesh Student HSC Result Predictor")
    gr.Markdown("Predict HSC exam results based on student information and background")
    
    with gr.Group():
        gr.Markdown("### Student Information")
        
        with gr.Row():
            gender = gr.Dropdown(
                choices=["M", "F"],
                label="Gender",
                value="M"
            )
            age = gr.Number(
                label="Age",
                value=18,
                minimum=15,
                maximum=30,
                precision=0
            )
        
        with gr.Row():
            address = gr.Dropdown(
                choices=["Rural", "Urban"],
                label="Address Type",
                value="Rural"
            )
            famsize = gr.Dropdown(
                choices=["LE3", "GT3"],
                label="Family Size",
                value="LE3",
                info="LE3: <=3 members, GT3: >3 members"
            )
        
        with gr.Row():
            pstatus = gr.Dropdown(
                choices=["Together", "Apart"],
                label="Parent Status",
                value="Together"
            )
            relationship = gr.Dropdown(
                choices=["Yes", "No"],
                label="In Relationship",
                value="No"
            )
        
        smoker = gr.Dropdown(
            choices=["Yes", "No"],
            label="Smoker",
            value="No"
        )
    
    with gr.Group():
        gr.Markdown("### Parent Education & Jobs")
        
        with gr.Row():
            m_edu = gr.Number(
                label="Mother's Education Level (0-4)",
                value=2,
                minimum=0,
                maximum=4,
                precision=0
            )
            f_edu = gr.Number(
                label="Father's Education Level (0-4)",
                value=2,
                minimum=0,
                maximum=4,
                precision=0
            )
        
        with gr.Row():
            m_job = gr.Dropdown(
                choices=["At_home", "Health", "Other", "Services", "Teacher", "Farmer", "Business"],
                label="Mother's Job",
                value="Other"
            )
            f_job = gr.Dropdown(
                choices=["At_home", "Health", "Other", "Services", "Teacher", "Farmer", "Business"],
                label="Father's Job",
                value="Other"
            )
    
    with gr.Group():
        gr.Markdown("### Academic & Financial Info")
        
        with gr.Row():
            tuition_fee = gr.Number(
                label="Tuition Fee (BDT)",
                value=30000,
                minimum=0,
                maximum=200000,
                precision=0
            )
            time_friends = gr.Number(
                label="Time Spent with Friends (hours/week)",
                value=3,
                minimum=0,
                maximum=24,
                precision=0
            )
        
        ssc_result = gr.Number(
            label="SSC Result (GPA)",
            value=3.5,
            minimum=0,
            maximum=5,
            precision=2
        )
    
    # Output
    output = gr.Markdown("### Prediction Result")
    
    # Predict button
    predict_btn = gr.Button("🔮 Predict HSC Result", size="lg", variant="primary")
    predict_btn.click(
        fn=predict_hsc_result,
        inputs=[
            gender, age, address, famsize, pstatus, m_edu, f_edu,
            m_job, f_job, relationship, smoker, tuition_fee,
            time_friends, ssc_result
        ],
        outputs=output
    )
    
    gr.Markdown("---")
    gr.Markdown("📊 **Model Info**: Random Forest Regressor trained on Bangladesh student performance data")

# =====================
# Launch app
# =====================
if __name__ == "__main__":
    demo.launch(share=True)