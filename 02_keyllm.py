documents = [
    "1:{how many tourists visit Chichester? how much rain falls on Chichester?}",
    "2: {Havant Thicket Resrvoir is going to include recycled waste water into the reservoir. Is this sensible using recycled water as drinking water? How much is water company earning? How much profit is southern water company earning every year? I don't think tourism should be stopped, such as airbnb. I think we should increase the trouism tax if it damages the environment so bad, the money can be used to protect environment, and farmers can use that money to build reservior on site to keep runoff from going too harbour. i know they are double the tax on second homes, and i like that, second homes should be housing, not vacant. but where does all the money go? and the southern water compnay is earning 37 million pound, they should be putting money into upgrading their waster water treatment works. }",
    "3: {Where will the money for investment in upgrading water treatment works come from? I don't think the money raised by the tax on second homes will be anywhere near sufficient.}",
    "4: {How can we prevent the water authorities (Southern Water ) from releasing bacteria, drugs and plastics into our harbours? sometimes the beachbuoy doesn't work. and the environment agency doesn't really work with local farmers to reduce nutrient runoff. }",
    "5: {how healthy is the oyster population in the itchenor channel}",
    "6: {what are solutions ffor the used fibreglass  boats in chichester and microplastics}",
    "7: {Hi, I am a house owner, I just bought an old house by del Quay and I want to rebuild. What conditions or policies do I have to adhere too? We also have a purple iris in our garden. Thank you! And if I have an owls nest in the garden? And a purple orchid patch? which area of chichester is especially protected? are there any great crested newts in the chichester area? }",
    "8: {what is the average household water use in Selsey? Where are the combined sewer overflows in Chichester Harbour? Where are flood risk areas in Chichester?}",
    "9: {What happens if one of the water authorities infringes the legislation? It seems that,apart from the occasional very large fine, the water authorities get away with breaking the law. what is storm overflows taskforce? how do i get in touch  with storm overflows taskforce?}",
    "10: {what are the protected wildlife in chichester? the distrabution of oyster bed? There is microplastic in the sludge. what measures can be take to get rid of microplastic so that sludge can be used as a cheap fertilizer for the farmers? I watched 'six inches of soil', and I don't think regulatory is working. Do you know what 'six inches of soil' is? And there is no consensus how water quality should be examined. I read something about they only check the water contamination on the surface, but the number is different down below. And do you know greenblue urban. they investigated tree species soil volumn, and found that underground utility is affecting tree planting. Near my living, somewhere around Chichester Festival Center, there is two trees that are same species and planted at the same time, but over the years, they are completed different, one of they seems to stop growing. I think our underground infrastructure like water pipes, is limiting this tree planting solution, nature based solution. And there is also flood new my home, those little streams. They flood sometime, its not mapped on Flood risk map of the government, but they flood. and the grass just becomes  wet dirt and sinky to walk on. To be specific, its around Platinum Jubilee Country Park and Bishop Luffa CofE Teaching School. What do you think is causing that? The streams are not as big as lavent, but they are there. and there are also constructions going on the stream - around Whitehouse Primary School, its new build. and they are right on top of these streams. sometimes the streams go dry, and flood and contaminated by construction. }",
]

import openai
from keybert.llm import OpenAI
from keybert import KeyLLM

# Create your LLM
OPENAI_API_KEY = ""
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# DEEPSEEK_API_KEY = ""
# client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
llm = OpenAI(client)

# Load it in KeyLLM
kw_model = KeyLLM(llm)

# Extract keywords
keywords = kw_model.extract_keywords(documents)
