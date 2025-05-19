# pylint: disable=C0115

from enum import Enum
from typing import List, Union, Literal, Dict, Any
from pydantic import BaseModel, Field


class VoiceOptions(str, Enum):
    ALLOY = "alloy"
    ECHO = "echo"
    ASH = "ash"
    CORAL = "coral"
    BALLAD = "ballad"
    SHIMMER = "shimmer"
    SAGE = "sage"
    VERSE = "verse"


class ClientSecret(BaseModel):
    value: str
    expires_at: int


class InputAudioTranscription(BaseModel):
    model: str = "whisper-1"


class TurnDetection(BaseModel):
    type: str = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = True


class ToolParameters(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Any]
    required: List[str]


class Tool(BaseModel):
    type: str = "function"
    name: str
    description: str
    parameters: ToolParameters


class Session(BaseModel):
    """
    Represents a session configuration for realtime voice chat.

    Attributes:
        modalities (List[str]): A list of modality names to be used in the session.
        instructions (str): Instructions or prompts provided for the session.
        voice (str): The identifier or description of the voice to be used.
        input_audio_format (str): The format of the input audio.
        output_audio_format (str): The format of the output audio.
        input_audio_transcription (InputAudioTranscription): An object that defines the settings for input audio transcription.
        turn_detection (TurnDetection): An object that manages turn detection parameters within the session.
        tools (List[Tool]): A list of tools available for use during the session.
        tool_choice (Literal["auto", "none", "required"]): Specifies the tool selection mode, with "auto" as the default.
        temperature (float): A value controlling randomness in output responses.
        max_response_output_tokens (Union[int, Literal["inf"]]): The maximum number of tokens allowed in the output response, or "inf" for no limit.
    """

    modalities: List[str] = Field(default_factory=lambda: ["text", "audio"])
    instructions: str = "You are a helpful assistant."
    voice: VoiceOptions = VoiceOptions.ALLOY
    input_audio_format: str = "g711_ulaw"
    output_audio_format: str = "g711_ulaw"
    input_audio_transcription: InputAudioTranscription = Field(
        default_factory=InputAudioTranscription
    )
    turn_detection: TurnDetection = Field(default_factory=TurnDetection)
    tools: List[Tool] = Field(default_factory=lambda: [])
    tool_choice: Literal["auto", "none", "required"] = "auto"
    temperature: float = 0.8
    max_response_output_tokens: Union[int, Literal["inf"]] = "inf"
