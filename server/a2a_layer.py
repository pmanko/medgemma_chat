import threading
import uuid
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional


class AgentRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._agents: Dict[str, Dict[str, Any]] = {}

    def register(self, agent_info: Dict[str, Any]) -> None:
        with self._lock:
            agent_id = agent_info.get("agent_id")
            if not agent_id:
                raise ValueError("agent_info must include 'agent_id'")
            self._agents[agent_id] = agent_info

    def lookup(self, task_name: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [info for info in self._agents.values() if info.get("task_name") == task_name]

    def list_agents(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._agents.values())


class MessageBus:
    def __init__(self) -> None:
        self._mailboxes: Dict[str, Queue] = {}
        self._lock = threading.RLock()

    def register_mailbox(self, agent_id: str) -> None:
        with self._lock:
            if agent_id not in self._mailboxes:
                self._mailboxes[agent_id] = Queue()

    def post_message(self, message: Dict[str, Any]) -> None:
        receiver_id = message.get("receiver_id")
        if not receiver_id:
            raise ValueError("message must include 'receiver_id'")
        with self._lock:
            if receiver_id not in self._mailboxes:
                self._mailboxes[receiver_id] = Queue()
            self._mailboxes[receiver_id].put(message)

    def get_message(
        self,
        agent_id: str,
        timeout: Optional[float] = None,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            mailbox = self._mailboxes.get(agent_id)
            if mailbox is None:
                mailbox = Queue()
                self._mailboxes[agent_id] = mailbox
        try:
            if predicate is None:
                return mailbox.get(timeout=timeout)
            # Poll loop with predicate support
            while True:
                msg = mailbox.get(timeout=timeout)
                if predicate(msg):
                    return msg
                # Not for us now; requeue at the end to avoid starvation
                mailbox.put(msg)
        except Empty:
            return None


def new_message(
    sender_id: str,
    receiver_id: str,
    task_name: str,
    payload: Dict[str, Any],
    status: str = "new",
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "message_id": str(uuid.uuid4()),
        "correlation_id": correlation_id or str(uuid.uuid4()),
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "task_name": task_name,
        "payload": payload,
        "status": status,
    }


# Global singletons
registry = AgentRegistry()
bus = MessageBus() 