import os
from shiny import App, ui, reactive, render
import httpx

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:5000")
API_URL = f"{API_BASE_URL}/predict"

# 1. UI
app_ui = ui.page_fluid(
    ui.h2("📰 Fake News Detector"),
    ui.p("Enter a title and article text, then press Predict."),

    ui.input_text("title", "Article Title:", ""),
    ui.input_text_area("text", "Article Text:",
                       "",
                       rows=10),

    ui.input_action_button("submit", "Predict", class_="btn-primary"),
    ui.hr(),
    ui.h4("Prediction Result"),
    ui.output_ui("result")
)



# 2. Server
def server(input, output, session):
    prediction_result = reactive.Value({"status": "ready", "message": "Awaiting input..."})

    #when the button is clicked
    @reactive.Effect
    @reactive.event(input.submit)
    async def _():
        #"Loading" message
        prediction_result.set({"status": "loading", "message": "Predicting..."})

        title = input.title()
        text = input.text()

        if not title or not text:
            prediction_result.set({"status": "error", "message": "❌ Please provide both a title and text."})
            return

        #async POST request
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    API_URL,
                    json={"title": title, "text": text},
                    timeout=20.0
                )

            #http status errors check
            if res.status_code != 200:
                prediction_result.set({
                    "status": "error",
                    "message": f"❌ API HTTP Error {res.status_code}. Response: {res.text}"
                })
                return

            # Success
            data = res.json()
            label_index = data.get("prediction_label_index")
            probs = data.get("prediction_probabilities")

            label = "FAKE NEWS" if label_index == 1 else "REAL NEWS"
            color = "text-danger" if label_index == 1 else "text-success"

            prediction_result.set({
                "status": "success",
                "label": label,
                "color": color,
                "probs": probs
            })

        except httpx.ConnectError:
            prediction_result.set({"status": "error",
                                   "message": f"🚨 Connection Error: Could not reach Flask API at {API_URL}. Is it running?"})
        except Exception as e:
            prediction_result.set({"status": "error", "message": f"🚨 Unknown Error: {str(e)}"})

    @output
    @render.ui
    def result():
        res = prediction_result.get()

        if res["status"] == "loading":
            return ui.p(res["message"])
        elif res["status"] == "error":
            return ui.p(res["message"], class_="text-danger fw-bold")
        elif res["status"] == "success":
            # Display results if successful
            prob_text = f"P(Real): {res['probs'][0]:.4f} | P(Fake): {res['probs'][1]:.4f}"
            return ui.div(
                ui.h3(res["label"], class_=f"{res['color']} fw-bold"),
                ui.p(prob_text, class_="text-muted")
            )
        else:
            return ui.p(res["message"])


#App Object
app = App(app_ui, server)
#terminal run command and access link
#& "C:\Program Files\Python314\python.exe" -m shiny run --reload app.py
#http://127.0.0.1:8000/