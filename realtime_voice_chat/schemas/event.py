# pylint: disable=C0115

from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field
from realtime_voice_chat.schemas.session import Session

# Add enum for event types
class EventType(str, Enum):
    SESSION_UPDATE = "session.update"
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    CREATE_CONVERSATION_ITEM = "conversation.item.create"
    CONVERSATION_ITEM_RETRIEVE = "conversation.item.retrieve"
    CONVERSATION_ITEM_DELETE = "conversation.item.delete"
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"


class BaseEvent(BaseModel):
    event_id: Optional[str] = None
    type: str


class SessionUpdateEvent(BaseEvent):
    type: EventType = Field(default=EventType.SESSION_UPDATE, literal=True)
    session: Session


class InputAudioBufferAppendEvent(BaseEvent):
    type: EventType = Field(default=EventType.INPUT_AUDIO_BUFFER_APPEND, literal=True)
    audio: bytes


class InputAudioBufferCommitEvent(BaseEvent):
    type: EventType = Field(default=EventType.INPUT_AUDIO_BUFFER_COMMIT, literal=True)


class InputAudioBufferClearEvent(BaseEvent):
    type: EventType = Field(default=EventType.INPUT_AUDIO_BUFFER_CLEAR, literal=True)


class ConversationItemContent(BaseModel):
    # Represents individual content pieces for a conversation item.
    # For role system: only input_text; for user: input_text and input_audio; for assistant: text.
    type: str
    text: str


class ConversationItem(BaseModel):
    id: str
    type: str
    role: str
    content: List[ConversationItemContent] = Field(default_factory=list)
    call_id: Optional[str] = Field(
        default=None
    )  # For function_call and function_call_output items.
    arguments: Optional[str] = Field(
        default=None
    )  # Arguments for the function call (for function_call items).


class ConversationItemCreateEvent(BaseEvent):
    type: str = Field(default=EventType.CREATE_CONVERSATION_ITEM, literal=True)
    previous_item_id: Optional[str]
    item: ConversationItem


class ConversationItemRetrieveEvent(BaseEvent):
    type: EventType = Field(default=EventType.CONVERSATION_ITEM_RETRIEVE, literal=True)
    item_id: str


class ConversationItemDeleteEvent(BaseEvent):
    type: EventType = Field(default=EventType.CONVERSATION_ITEM_DELETE, literal=True)
    item_id: str


class ResponseCreateEvent(BaseEvent):
    type: EventType = Field(default=EventType.RESPONSE_CREATE, literal=True)
    response: dict


class ResponseCancelEvent(BaseEvent):
    type: EventType = Field(default=EventType.RESPONSE_CANCEL, literal=True)
