from openai import OpenAI
import base64

from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def analyze_report(uploaded_file, notes=""):

    image_bytes = uploaded_file.read()

    base64_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    response = client.chat.completions.create(

        model="google/gemini-3.1-flash-lite-image",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Analyze the uploaded laboratory report.

This is an educational health-report explanation request.

Do NOT diagnose any disease.
Do NOT prescribe treatment.
Do NOT refuse simply because the report belongs to a real person.

Your job is to explain the report in simple language.

Additional Information From User:
{notes}

Use BOTH:
1. The laboratory report
2. The user's notes/symptoms

When generating the explanation.

For every test value:
- Compare it with the reference range shown in the report.
- Identify values that are normal.
- Identify values that are outside the reference range.
- Explain what abnormal values generally mean.
- Mention whether the user's symptoms may be related to any findings.

Keep the explanation friendly, simple, and easy for non-medical users.

Return ONLY in this exact format:

SUMMARY:
(2-3 sentences summarizing the report)

GOOD FINDINGS:
- finding 1
- finding 2
- finding 3

NEEDS ATTENTION:
- finding 1
- finding 2
- finding 3

SUGGESTIONS:
- suggestion 1
- suggestion 2
- suggestion 3

DISCLAIMER:
This is an educational explanation and not a medical diagnosis. Please consult a qualified healthcare professional for medical advice.

ONLY write in English.
Do not use markdown.
Do not use headings other than the ones above.
Do not write long paragraphs.
Keep everything patient-friendly.
If all values are normal, clearly say so.
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content