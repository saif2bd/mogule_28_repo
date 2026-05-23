import gradio as gr
import pandas as pd
import numpy as np
import pickle



# load model
with open("student_rf_pipeline.pkl", "rb") as file:
    model = pickle.load(file)



#main logic
def predict_gpa(gender, age, address, famsize,
                Pstatus, M_Edu, F_Edu, M_Job, F_Job,
                relationship, smoker, tuition_fee, time_friends,
                ssc_result):

    # Create dataframe
    input_data = pd.DataFrame(
        [[
            gender, age, address, famsize, Pstatus,
            M_Edu, F_Edu, M_Job, F_Job,
            relationship, smoker, tuition_fee,
            time_friends, ssc_result
        ]],

        columns=[
            'gender', 'age', 'address', 'famsize',
            'Pstatus', 'M_Edu', 'F_Edu',
            'M_Job', 'F_Job', 'relationship',
            'smoker', 'tuition_fee',
            'time_friends', 'ssc_result'
        ]
    )

    # Predict
    prediction = model.predict(input_data)[0]

    # Return clipped GPA
    return float(np.clip(prediction, 0, 5))

inputs = [
    gr.Radio(['M', 'F'], label = 'Gender'),
    gr.Number(label = 'Age', value = 18),
    gr.Radio(["Urban", "Rural"], label = 'Address'),
    gr.Radio(["LE3", "GT3"], label = 'Family Size'),
    gr.Radio(["Togather", "Apart"], label = 'Parental Status'),
    gr.Slider(0, 4, step=1, label = 'Mother Education Level'),
    gr.Slider(0, 4, step=1, label = 'Father Education Level'),
    gr.Dropdown(["At_home", "Health", "Other", "Services", "Teacher"], label="Mother's Job"),
    gr.Dropdown(["Teacher", "Other", "Services", "Health", "Business", "Farmer"], label="Father's Job"),
    gr.Radio(["Yes", "No"], label="Relationship"),
    gr.Radio(["Yes", "No"], label="Smoker"),
    gr.Number(label="Tuition Fee"),
    gr.Slider(1, 5, step=1, label="Time with Friends"),
    gr.Number(label="SSC Result (GPA)")

]

#interface
app = gr.Interface(
    fn = predict_gpa,
    inputs = inputs,
    outputs = 'text',
    title = "HSC Result Predictor"
)


#launch
app.launch(share=True)
