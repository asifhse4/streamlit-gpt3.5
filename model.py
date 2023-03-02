import openai

poem = """Write a poem with the following words: 
---
{input}
---
This is the poem: """

start_sequence = "\nOJO:"
restart_sequence = "\n\nUser:"


personality_prompt = """
    You are OJO, a GPT3 Chat bot assistant for the Canadian real estate market.
    OJO knows a lot about active homes for sale, agents, listing details, real estate in general, and real estate Market conditions in Canada.
    You can ask OJO questions about real estate and will you get a helpful, knowledgeable answer.  OJO will admit when it does not know the answer.
    OJO will answer all questions with the style of an experienced real estate agent. Long sentences are ok.
    OJO will try and and answer from the listing context below, and admit when you do not know the answer.
    When you don't know the answer to a User question, you will reply "Sorry, I don't have the data to answer that."
    When OJO does not know the answer to a User question, OJO will reply "Sorry, I don't have the data to answer that."
    """


listing_prompt = """
    The home at the address 50 Candle Ter SW Calgary, AB T2W 6G7 is for sale for $550,000.00.
    
    This home is 29% Below the Calgary neighbourhood market average price.
    
    This home has been on the market for 7 days.
    
    There are no other homes like this on sale in this area.
    
    This home was built in 1996.  This home is 27 years old.
    
    The lot size for this home is 2,614 square feet.
    
    The living room dimensions are 16 feet by 22 feet.
    
    The dimensions for this garage are unknown.
    
    This home is one of 13 for sale in Canyon Meadows. The median days on the market for homes in this area is 11 with a median list price of $800,000.00 and median cost per square foot of $346. 
    The list price of this home is 31% below the Canyon Meadows median, It has a price per square foot of $386, which is 11% above the median and has been on the market for 1 day.
    
    Schools for this home's neighborhood are rated 7.3/10 Data computed based on school score and ranking provided by the Fraser Institute.
    
    Within the Calgary real estate market, 3.83% of Home prices in Calgary have increased since May, 2022.
    
    Nearby Points of Interest to this home are:
        There is a Resturant nearby called Filipino Cafe.  Filipino Cafe is 1.40 kilometers away. 
        There is a Grocery Store called Husky that is 0.20 kilometers away.
        There is a park called Lake Bonavista Park nearby. Lake Bonavista Park is 3.60 kilometers away.
        There is a Resturant nearby.  The resturant is called Creekside Bar & Grill that is 0.20 kilometers away.

    The features of this house are Bungalow, Hunter douglas blinds, Dining area, Laundry area, Deck, Laundry room, Living room, Cabinet space, Basement, Staycation ready, Gas fireplace, Stucco facade, Water softener, Walk-in closet, RenovatedFireplace, Perfect for pets, Main floor laundry room, Vinyl plank floors, Move-in ready, High ceilings, Backyard, Vaulted ceilings, Composite deck, Completely remodeled, Flex room, Formal dining room, Eco-friendly, Single-level home, Gas heating, Full basement, Attic, Accessible home, Perfect for kids, Green thumb required, For the young professional, Loft space, Perfect for families, Updated & renovated, and Wide open spaces.
    
    This home features a Bungalow, Hunter douglas blinds, Dining area, Laundry area, and a Deck. 
"""


agent_prompt = """
    The agent for this home is Art Vandelay.  
    Art has been an agent for 9 years, and has been working with OJO for 9 months. 
    Art has experience selling this type of home.
    In the last 12 months, Art has sold 4 homes at an average price of $500,00.00
"""


session_prompt = f"""
    {personality_prompt}
    
    {agent_prompt}
    
    Listing context:
    {listing_prompt}
    
    \n\nUser: Does this home have a pool?\nOJO: Yes, this home has a pool.
    \n\nUser: How many bathrooms does it have? \nOJO: This home has 3 bathrooms.  
    \n\nUser: Do homes like this tend to sell fast? \nOJO:Homes like this one in this area tend to be on the market for an average of 34 days.  What else can I help you with?
    \n\nUser: How tall is the garage? \nOJO: Sorry, I don't have the data to answer that.
    \n\nUser: Has the house ever been flooded? \nOJO: Sorry, I don't have the data to answer that.
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
    
    prompt_text = f"{chat_log}{restart_sequence} {question}{start_sequence}"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_text,
        temperature=0.1,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
     )["choices"][0]["text"].strip(" \n")
    
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

        
        prompt_text = f"{session_prompt}{restart_sequence} {prompt}{start_sequence}"

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt_text,
            temperature=0.7,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
         )["choices"][0]["text"].strip(" \n")
        
        
        
#         # arguments to send the API
#         kwargs = {
#             "engine": "davinci",
#             "temperature": 1,
#             "max_tokens": 300,
#             "top_p": 1,
#             "frequency_penalty": 0,
#             "presence_penalty": 0,
#             "stop": ["\n"],
#         }

#         for kwarg in myKwargs:
#             kwargs[kwarg] = myKwargs[kwarg]

#         print('Prompt Text:', prompt_text)
            
#         r = openai.Completion.create(prompt=prompt_text, **kwargs)["choices"][0]["text"].strip(" \n")
        
        return response

    def model_prediction(self, input, api_key):
        """
        wrapper for the API to save the prompt and the result
        """
        # Setting the OpenAI API key got from the OpenAI dashboard
        set_openai_key(api_key)
        output = self.query(input)
        return output