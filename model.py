import openai

poem = """Write a poem with the following words: 
---
{input}
---
This is the poem: """

start_sequence = "\nOJO:"
restart_sequence = "\n\nUser:"


personality_prompt = """
    You are talking to OJO, GPT3 Chat bot assistant for the Canadian real estate market.
    OJO knows a lot about active homes for sale, agents, listing details, real estate in general, and real estate Market conditions in Canada.
    You can ask OJO questions about real estate and will you get a helpful, knowledgeable answer.  
    OJO will not make up answers to questions that it does not know.  OJO will admit when it does not know the answer.
    
    """


listing_prompt = """
    The home at the address 50 Candle Ter SW Calgary, AB T2W 6G7 is for sale for $550,000.00.
    
    This home ahs 3 bedrooms and 3 bathrooms.  It has 1,400 square feet of space.
    
    This home is 29% Below the Calgary neighbourhood market average price.
    
    This home has been on the market for 7 days.
    
    There are a no other homes like this on sale in this area.
    
    Schools for this home's neighborhood are rated 7.3/10 Data computed based on school score and ranking provided by the Fraser Institute.
    
    Within the Calgary real estate market, 3.83% of Home prices in Calgary have increased since May, 2022.
    
    Nearby points of interest to this home are Cafe Filipino at 1.40 Km away. 
    There is a Grocery Store called Husky that is 0.20 Km away.  
    There is a park called Lake Bonavista Park that is 3.60 Km away
    There is a Resturant called Creekside Bar & Grill that is 0.20 Km away.
    There is a park nearby called Lake Bonavista Park. This park is 3.60 Km away.
    
    OJO does not know when this house was built.

    The features of this house are Bungalow, Hunter douglas blinds, Dining area, Laundry area, Deck, Laundry room, Living room, Cabinet space, Basement, Staycation ready, Gas fireplace, Stucco facade, Water softener, Walk-in closet, RenovatedFireplace, Perfect for pets, Main floor laundry room, Vinyl plank floors, Move-in ready, High ceilings, Backyard, Vaulted ceilings, Composite deck, Completely remodeled, Flex room, Formal dining room, Eco-friendly, Single-level home, Gas heating, Full basement, Attic, Accessible home, Perfect for kids, Green thumb required, For the young professional, Loft space, Perfect for families, Updated & renovated, and Wide open spaces.

    Bungalow is a feature of this house.

"""


agent_prompt = """
    The agent for this home is Art Vandelay.  
    Art has been an agent for 9 years, and has been working with OJO for 9 months. 
    Art has experience selling this type of home.
    In the last 12 months, Art has sold 4 homes at an average price of $500,00.00
"""


session_prompt = f"""
    {personality_prompt}
    {listing_prompt}
    {agent_prompt}
    \n\nUser: What is the value of this home?\nOJO: This home is being sold for $550,000.00.
    \n\nUser: How many bathrooms does it have? \nOJO: This home has 3 bathrooms.
    \n\nUser: Do homes like this tend to sell fast? \nOJO: Homes like this one in this area tend to be on the market for an average of 34 days.
    """


def append_interaction_to_chat_log(
    question: str, 
    answer: str, 
    session_prompt: str=session_prompt, 
    chat_log: str=None):
    """
    question: previous question in the conversation
    answer: previous answer in the conversation
    session_prompt: Chat bot personality that is set once if no chat log is present
    chat_log: prior chat logs
    """
    if chat_log is None:
        return session_prompt 
    return f'{chat_log}{question}{start_sequence}{answer}'

def ask(question: str, chat_log: str=None):
    """
    Ask GPT a question using a chat log with session prompt including
    """
    
    prompt_text = f"{chat_log}{restart_sequence}: {question}{start_sequence}:"

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt_text,
        temperature=1,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
     )["choices"][0]["text"].strip(" \n")
    
    print(response.split('User:'))
    print(response)
    
    return response



def set_openai_key(key):
    """Sets OpenAI key."""
    openai.api_key = key

class GeneralModel:
    def __init__(self):
        print("Model Intilization--->")
        #self.chat_log = append_interaction_to_chat_log('How much is this house?', '', session_prompt=session_prompt, chat_log=None)
        # set_openai_key(API_KEY)

    def query(self, prompt, myKwargs={}):
        """
        wrapper for the API to save the prompt and the result
        """

        prompt_text = f"{session_prompt}{restart_sequence}: {prompt}{start_sequence}:"
        
        # arguments to send the API
        kwargs = {
            "engine": "davinci",
            "temperature": 1,
            "max_tokens": 300,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": ["\n"],
        }

        for kwarg in myKwargs:
            kwargs[kwarg] = myKwargs[kwarg]

        print('Prompt Text:', prompt_text)
            
        r = openai.Completion.create(prompt=prompt_text, **kwargs)["choices"][0]["text"].strip(" \n")
        
        return r

    def model_prediction(self, input, api_key):
        """
        wrapper for the API to save the prompt and the result
        """
        # Setting the OpenAI API key got from the OpenAI dashboard
        set_openai_key(api_key)
        output = self.query(input)
        return output