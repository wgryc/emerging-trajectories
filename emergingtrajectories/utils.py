from phasellm.llms import OpenAIGPTWrapper, ChatBot

_extract_prediction_prompt = """You are helping a researcher with a data extraction exercise. You will be provided with a prediction statement and a broader piece of text. Your objective is to extract the specific numerical prediction and provide it as a response. DO NOT qualify your response in any way.

For example, suppose you have the following...

|||---|||
PREDICTION STATEMENT: please predict the probability that the price of Bitcoin will exceed $100,000 by the end of 2024.

TEXT: The probability that bitcoin will exceed $100,000 by the end of 2024 is 0.37.
|||---|||

In the case above, your response would simply be "0.37".

The actual metrics (i.e., prediction) might be provided with formatting. For example...

|||---|||
PREDICTION STATEMENT: The probability that Boeing's (NYSE:BA) share price at the close of markets on or before March 1, 2024 will be $220.00 USD or higher is _____ (value between 0.00 and 1.00).

TEXT: The probability that Boeing's (NYSE:BA) share price at the close of markets on or before March 1, 2024, will be $220.00 USD or higher is **0.65**.
|||---|||

In this case, ignore the asterisks or instructions ("value between 0.00 and 1.00") and provide the correct response, which is 0.65.

The user will provide you with a PREDICTION STATEMENT and TEXT and you need to answer like the above.

On the extremely rare occasion that the TEXT does not have a proper numerical prediction or you are unable to extract it, simply respond with "UNCLEAR".
"""

statement_1 = """Trump will beat Haley in the 2024 New Hampshire Republican primary by ___ percentage points."""

text_1 = """Based on the information provided and the current political climate, my prediction for the 2024 New Hampshire Republican primary is as follows:

- Donald Trump: 45%
- Nikki Haley: 35%

In other words, Trump will beat Haley by 10%.

Justification:

1. Polling Data: The most recent CNN and the University of New Hampshire poll shows Trump leading with 39% and Haley at 32%. Given Haley's upward trajectory, gaining 12 percentage points since November, and Trump's slight decline, it's reasonable to predict a further narrowing of the gap. However, Trump's strong national polling suggests he will maintain a lead.

2. Iowa Caucus Results: Trump's significant win in the Iowa Caucus, as reported by various sources, indicates strong support within the Republican base. This momentum is likely to carry over into New Hampshire, although the impact may be lessened due to New Hampshire's more moderate and independent electorate.

3. Independent Voters: Haley's appeal to unaffiliated voters, who have fueled her rise in New Hampshire, suggests she will perform well. However, Trump's lead among conservatives and the fact that he still holds a commanding lead nationally suggests he will retain a significant portion of the vote.

4. Favorability: Both Trump and Haley have positive favorability scores among likely GOP primary voters, but Trump's higher favorability suggests he will maintain an edge.

5. Campaign Dynamics: Haley's focused campaign in New Hampshire and the potential impact of her endorsements may boost her performance. However, Trump's established base and the inertia from his previous presidency provide him with a strong foundation that is likely to keep him ahead.

6. Historical Trends: Incumbent presidents or high-profile figures like Trump often perform well in primaries due to name recognition and established support networks, which gives him an advantage.

7. Weather and Turnout: The extreme cold during the Iowa caucuses may have suppressed turnout, which could have favored Trump's dedicated base. New Hampshire's weather may not be as severe, potentially allowing for a better turnout among a more diverse set of voters, which could benefit Haley.

8. National Mood: The current political climate, with a focus on strong leadership and economic concerns, may favor Trump's messaging and perceived strengths among the Republican electorate.

Taking all these factors into account, while Haley is likely to improve her standing and close the gap with Trump, the incumbent advantage, national polling, and Trump's strong showing in Iowa suggest he will maintain a lead over Haley in the New Hampshire Republican primary."""

_extract_prediction_prompt_error = "UNCLEAR"

def is_numeric(string):
    if string is None:
        return False
    try:
        float(string)
        return True
    except ValueError:
        return False

class UtilityHelper(object):

    def __init__(self, api_key, model="gpt-4-0125-preview"):
        self.api_key = api_key
        self.model = model

    def extract_prediction(self, response, statement_challenge):

        message_stack = [
            {"role":"system", "content": _extract_prediction_prompt},
            {"role":"user", "content": f"PREDICTION STATEMENT: {statement_challenge}\n\nTEXT: {response}"},
        ]

        #print(f"PREDICTION STATEMENT: {statement_challenge}\n\nTEXT: {response}")

        llm = OpenAIGPTWrapper(apikey=self.api_key, model=self.model)
        chatbot = ChatBot(llm)
        chatbot.messages = message_stack
        
        output = chatbot.resend()

        print(f"\n\n\n{output}\b\b\b")

        if output == _extract_prediction_prompt_error:
            raise Exception("Unable to extract prediction from response.")

        if output[0] == "$":
            output = output[1:]

        #print(output)

        if not is_numeric(output):
            raise Exception(f"Prediction does not appear to be numeric:\n{output}")
        
        return float(output)