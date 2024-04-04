from phasellm.llms import OpenAIGPTWrapper, ChatBot

# Prompt used for extracting predictions from text messages.
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

# Error message used when the prediction cannot be extracted from the response.
_extract_prediction_prompt_error = "UNCLEAR"


def is_numeric(string: str) -> bool:
    """
    Checks whether the 'string' passed as an argument can be converted into a numeric value.

    Args:
        string: the string in question

    Returns:
        Boolean value; True if the string can be converted into a numeric value, False otherwise.
    """
    if string is None:
        return False
    try:
        float(string)
        return True
    except ValueError:
        return False


# TODO document
def run_forecast(function_to_call, n, *args, **kwargs):

    if n == 0:
        return None

    result = None

    try:
        result = function_to_call(*args, **kwargs)
    except Exception as e:
        print(f"Forecast failed with error: {e}")
        print(f"Trying up to {n-1} more times.")
        result = run_forecast(function_to_call, n - 1, *args, **kwargs)

    if result is None:
        print(f"Forecast failed after {n} attempts.")

    return result


class UtilityHelper(object):

    def __init__(self, api_key, model="gpt-4-0125-preview") -> None:
        """
        The UtilityHelper class is used to extract predictions from text messages.

        Args:
            api_key: the OpenAI API key
            model: the OpenAI model to use for the extraction process
        """

        self.api_key = api_key
        self.model = model

    def extract_prediction(self, response: str, statement_challenge: str) -> float:
        """
        Extracts the prediction value from the response to a statement challenge.

        Args:
            response: the response to the statement challenge (i.e., what was predicted by another LLM)
            statement_challenge: the statement challenge -- what is being predicted

        Returns:
            The extracted prediction value as a float. Raises an exception if the prediction cannot be extracted.
        """

        message_stack = [
            {"role": "system", "content": _extract_prediction_prompt},
            {
                "role": "user",
                "content": f"PREDICTION STATEMENT: {statement_challenge}\n\nTEXT: {response}",
            },
        ]

        # print(f"PREDICTION STATEMENT: {statement_challenge}\n\nTEXT: {response}")

        llm = OpenAIGPTWrapper(apikey=self.api_key, model=self.model)
        chatbot = ChatBot(llm)
        chatbot.messages = message_stack

        output = chatbot.resend()

        # print(f"\n\n\n{output}\b\b\b")

        if output == _extract_prediction_prompt_error:
            raise Exception("Unable to extract prediction from response.")

        if output[0] == "$":
            output = output[1:]

        # Remove commas...
        output = output.replace(",", "")

        if not is_numeric(output):
            raise Exception(f"Prediction does not appear to be numeric:\n{output}")

        return float(output)
