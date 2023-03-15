from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.codecs import NumpyCodec


COMPLEX_CLAIMS_PAYOUT_RATE = 0.6

class CalculateComplexClaimPayout(MLModel):

    async def load(self) -> bool:
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        amount_input = next(x for x in payload.inputs if x.name == "total_claim_amount")
        amount_value = NumpyCodec.decode_input(amount_input)
        payout_value = amount_value * COMPLEX_CLAIMS_PAYOUT_RATE

        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                NumpyCodec.encode_output("claim_payout", payout_value)
            ]
        )