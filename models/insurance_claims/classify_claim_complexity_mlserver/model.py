from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.codecs import NumpyCodec, StringCodec, InputCodec
import numpy as np
from typing import Type


SIMPLE_CLAIM_VALUE_THRESHOLD = 10000

class ClassifyClaimComplexity(MLModel):

    async def load(self) -> bool:
        self.ready = True
        return self.ready

    def get_input_value(self, payload: InferenceRequest, input_name: str, codec: Type[InputCodec]):
        input = next(x for x in payload.inputs if x.name == input_name)
        print(input)
        input_numpy = codec.decode_input(input)
        return input_numpy[0]

    def is_claim_complex(self, payload: InferenceRequest):
        claim_amount = self.get_input_value(payload, "total_claim_amount", NumpyCodec)
        if claim_amount <= SIMPLE_CLAIM_VALUE_THRESHOLD:
            # small claims are never complex
            return False

        auto_year = self.get_input_value(payload, "auto_year", NumpyCodec)
        if auto_year < 2000:
            # old cars yield complex cases
            return True
        
        witnesses = self.get_input_value(payload, "witnesses", NumpyCodec)
        police_report_available = self.get_input_value(payload, "police_report_available", StringCodec)
        if witnesses == 0 and police_report_available != "YES":
            # no objective evidence of incident cause
            return True

        return False

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        if self.is_claim_complex(payload):
            output_name = "is_complex_claim"
        else:
            output_name = "is_simple_claim"
        output_data = np.ones((1), dtype=bool)

        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                NumpyCodec.encode_output(output_name, output_data)
            ]
        )