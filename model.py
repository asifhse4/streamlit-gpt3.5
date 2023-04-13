import openai

start_sequence = "\nOJO:"

restart_sequence = "\n\nUser:"

listing_address = "377 Sackville St, Toronto, ON, M5A 3G5, Canada"

personality_prompt = f""" Your name is OJO. You are an assistant for the Canadian real estate market. Reply to user questions in the same language that the user asks. Only reply questions related to the property at {listing_address} from information below. When data is not available, or if you don't know the answer or you're not confident then you will redirect the user to contact an agent by filling out the "Connect with Agent" form on https://www.ojohome.ca/. For all questions related to price, mortgages, pre-qualification, pre-approval and finance, answer the question and then refer the user to webpage on mortages https://www.ojohome.ca/mortgages/preapproved-vs-prequalified/. If the user is interested in home or want to buy then refer to fill the contact agent form on the page the user is on or "Connect with Agent" form on https://www.ojohome.ca/. For questions regarding removing listings, or anything non-transactional, direct the user to the help center https://help.ojohome.ca/hc/en-us?utm_source=footer&utm_medium=link, or call OJO at 1-833-709-1946. Never comment on good or bad investment, answer it by asking them to contact an agent by filling out the "Connect with Agent" form on https://www.ojohome.ca/. Never show additional photos or email anyone. Never try to reach out or email a user. Never connect the user to an agent, but the user can fill out an agent contact form. Only answer questions from the information provided in the user content. Do not call yourself as AI. Do not link walkable score with crime. Do not start by replying "I'm sorry". """


listing_prompt = """{'property posted on': '2023-04-05 23:51:38', 'days on market': 6, 'mls name': 'TREB', 'mls number': 'C6006927', 'home type': 'Single Family', 'property type': 'Detached', 'property description': 'Cannot be overstated! truly one of a kind french empire majestic home c. 1876, detached 5+1 br house on one of a kind double lot, recently restored and renovated to the highest level of finish! private drive with detached garage, located in the coveted sprucecourt school district, short walk to great shops & restaurants on parliament', 'street name': 'Sackville', 'fsa': 'M5A', 'city': 'Toronto', 'state': 'ON', 'zipcode': 'M5A 3G5', 'community': 'Cabbagetown-south St. James Town', 'neighborhood': 'Cabbagetown', 'latitude': 43.664491, 'longitude': -79.364495, 'agent name': '', 'agent license number': '', 'current listing price': 4298000, 'original listing price': 4298000, 'price changed': None, 'tax amount': 17359, 'tax year': 2022, 'house area in sqft': None, 'bedrooms': 6, 'bedrooms above grade': 5, 'bedrooms below grade': 1, 'bedrooms on grade': None, 'bathroom': 5.0, 'partial bathroom': None, 'garage': 1, 'parking': 2.0, 'lot size': 5472, 'built year': None, 'basement type': None, 'in demand home': False, 'pool': None, 'property architecture style': '3-Storey', 'List Nearby schools, parks or any Points of interest': 'Please refer to the Nearby Points of Interest section on this page', 'unique features': 'renovated, brick, garage', 'compare local market price': '73% more than local average price', 'price summary': 'This property is one of 13 for sale in Cabbagetown. The median days on the market for properties in this area is 6 with a median list price of $1.6M. The list price of this property is 168% above the Cabbagetown median, and has been on the market for 6 days.', 'median list price in Cabbagetown, Toronto, ON': '1.6 million CAD', 'median days on market in Cabbagetown, Toronto, ON': '6 days', 'properties for sale in Cabbagetown, Toronto, ON': 13, 'properties for sale in Toronto, ON': 2064, 'median days on market in Toronto, ON': '275 days', 'Walkable score - Shopping Center': 'Somewhat Walkable', 'Walkable score - Schools': 'Somewhat Walkable', 'Walkable score - Parks': 'Very Walkable', 'Walkable score - Restaurants': 'Very Walkable', 'school score': '6.8 out of 10'}"""

model = "gpt-3.5-turbo" 


def set_openai_key(key):
    """Sets OpenAI key."""
    openai.api_key = key
    
    
    

class GeneralModel:
    def __init__(self):
        print("Model Intilization--->")
        
        self.personality_prompt_ = personality_prompt
        self.listing_prompt_ = listing_prompt
        self.model = model
    
    def query(self, question_): 
        
        results = {} 
        
        response = openai.ChatCompletion.create( 
            model=self.model, 
            temperature=0.1, 
            messages=[
                
                {"role": "system", "content": self.personality_prompt_}, 
                {"role": "system", "content": "Only answer questions from the information provided in the user content"},
                {"role": "system", "content": "Do not answer any questions realted to nearby places or points of interest, redirect them to the Nearby Points of Interest section on this page"},
                {"role": "user", "content": f"{self.listing_prompt_}\n {question_}"} 
            
            ],
                      
             
        results["response"] = response["choices"][0]["message"]["content"] 
        results["role"] = response["choices"][0]["message"]["role"] 
        results["total_tokens"] = response["usage"]["total_tokens"] 
        results["prompt_tokens"] = response["usage"]["prompt_tokens"] 
        results["completion_tokens"] = response["usage"]["completion_tokens"] 
        results["model"] = response["model"] 
        results["object"] = response["object"] 
            
        return results["response"]

    def model_prediction(self, input, api_key):
        """
        wrapper for the API to save the prompt and the result
        """
        # Setting the OpenAI API key got from the OpenAI dashboard
        set_openai_key(api_key)
        output = self.query(input)
        return output