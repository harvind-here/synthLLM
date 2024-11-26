import os
import json
import random
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pandas as pd

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

class MedicalBilling(BaseModel):
    patient_id: int = Field(description="The unique identifier for the patient")
    patient_name: str = Field(description="The full name of the patient")
    diagnosis_code: str = Field(description="The ICD-10 diagnosis code")
    procedure_code: str = Field(description="The CPT procedure code")
    total_charge: float = Field(description="The total charge for the medical service")
    insurance_claim_amount: float = Field(description="The amount claimed from insurance")

parser = PydanticOutputParser(pydantic_object=MedicalBilling)

# Load the real dataset
with open("medical_billing_dataset.json", "r") as f:
    real_dataset = json.load(f)

# Create a string representation of the real dataset for the prompt
real_data_str = json.dumps(real_dataset, indent=2)

prompt = PromptTemplate(
    template="""You are tasked with generating synthetic medical billing data based on the following real data examples:

{real_data}

Generate a single instance of synthetic medical billing data that follows the patterns and distributions of the real data. Return only the JSON data without any explanation or code. 

{format_instructions}

Ensure that your output is a well-formatted JSON instance that conforms to the schema above. Do not include any schema information (like "properties" or "required") in your output. Your output should be a single JSON object with the specified fields directly at the top level.

{query}""",
    input_variables=["query", "real_data"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_synthetic_data(n=30):
    results = []
    print("Generating Dataset...")
    for _ in range(n):
        query = "Generate a single instance of synthetic medical billing data that closely resembles the real data examples."
        result = chain.run(query=query, real_data=real_data_str)
        try:
            # Try to parse the result as JSON
            json_result = json.loads(result)
            # Validate that all required fields are present
            if all(field in json_result for field in MedicalBilling.__fields__):
                results.append(MedicalBilling(**json_result))
            else:
                print(f"Generated data missing required fields: {result}")
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {result}")
        except Exception as e:
            print(f"Error processing result: {e}\nResult: {result}")
    return results

synthetic_results = generate_synthetic_data()

# Create a list of dictionaries from the objects
synthetic_data = [item.dict() for item in synthetic_results]

# Create a Pandas DataFrame from the list of dictionaries
synthetic_df = pd.DataFrame(synthetic_data)

print(synthetic_df)

# Save the generated dataset to a JSON file
with open("gen_dataset.json", "w") as f:
    json.dump(synthetic_data, f, indent=2)

print("Generated dataset saved to gen_dataset.json")
print('________________________________________________________________________________________________________')
print(parser.get_format_instructions())