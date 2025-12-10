from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

import dspy
import opik
from opik import Opik, config
from opik.integrations.dspy.callback import OpikCallback
from typing_extensions import Concatenate, ParamSpec

from .utils import validate_project_name

P = ParamSpec("P")  # params of the function
R = TypeVar("R")  # return type of the function
T = TypeVar("T", bound="Trackable")  # an instance of a class that inherits from Trackable


@dataclass
class ModaicSettings:
    """Global settings for Modaic observability."""

    tracing: bool = False
    project: Optional[str] = None
    base_url: str = "https://api.modaic.dev"
    modaic_token: Optional[str] = None
    default_tags: Dict[str, str] = field(default_factory=dict)
    log_inputs: bool = True
    log_outputs: bool = True
    max_input_size: int = 10000
    max_output_size: int = 10000


# global settings instance
_settings = ModaicSettings()
_opik_client: Optional[Opik] = None
_configured = False


def configure(
    project: str,
    tracing: bool = True,
    base_url: str = "https://api.modaic.dev",
    modaic_token: Optional[str] = None,
    default_tags: Optional[Dict[str, str]] = None,
    log_inputs: bool = True,
    log_outputs: bool = True,
    max_input_size: int = 10000,
    max_output_size: int = 10000,
    **opik_kwargs,
) -> None:
    """Configure Modaic observability settings globally.

    Args:
        tracing: Whether observability is enabled
        project: Default project name
        base_url: Opik server URL
        modaic_token: Authentication token for Opik
        default_tags: Default tags to apply to all traces
        log_inputs: Whether to log function inputs
        log_outputs: Whether to log function outputs
        max_input_size: Maximum size of logged inputs
        max_output_size: Maximum size of logged outputs
        **opik_kwargs: Additional arguments passed to opik.configure()
    """
    global _settings, _opik_client, _configured

    # update global settings
    _settings.tracing = tracing
    _settings.project = project
    _settings.base_url = base_url
    _settings.modaic_token = modaic_token
    _settings.default_tags = default_tags or {}
    _settings.log_inputs = log_inputs
    _settings.log_outputs = log_outputs
    _settings.max_input_size = max_input_size
    _settings.max_output_size = max_output_size

    if tracing:
        # configure Opik
        opik_config = {"use_local": True, "url": base_url, "force": True, "automatic_approvals": True, **opik_kwargs}

        opik.configure(**opik_config)

        _opik_client = Opik(host=base_url, project_name=project)
        opik_callback = OpikCallback(project_name=project, log_graph=False)
        dspy.configure(callbacks=[opik_callback])

    config.update_session_config("track_disable", not tracing)

    _configured = True


def _get_effective_settings(project: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Get effective settings by merging global and local parameters."""
    effective_project = project if project else _settings.project

    # validate project name if provided
    if effective_project:
        validate_project_name(effective_project)

    # merge tags
    effective_tags = {**_settings.default_tags}
    if tags:
        effective_tags.update(tags)

    return {"project": effective_project, "tags": effective_tags}


def _truncate_data(data: Any, max_size: int) -> Any:
    """Truncate data if it exceeds max_size when serialized."""
    try:
        import json

        serialized = json.dumps(data, default=str)
        if len(serialized) > max_size:
            return f"<Data truncated: {len(serialized)} chars>"
        return data
    except Exception:
        # if serialization fails, convert to string and truncate
        str_data = str(data)
        if len(str_data) > max_size:
            return str_data[:max_size] + "..."
        return str_data


def track(  # noqa: ANN201
    name: Optional[str] = None,
    project: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    span_type: str = "general",
    capture_input: Optional[bool] = None,
    capture_output: Optional[bool] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **opik_kwargs,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to track function calls with Opik.

    Args:
        name: Custom name for the tracked operation
        project: Project name (overrides global setting)
        tags: Additional tags for this operation
        span_type: Type of span ('general', 'tool', 'llm', 'guardrail')
        capture_input: Whether to capture input (overrides global setting)
        capture_output: Whether to capture output (overrides global setting)
        metadata: Additional metadata
        **opik_kwargs: Additional arguments passed to opik.track
    """

    def decorator(func: Callable) -> Callable:
        if not _settings.tracing:
            return func

        # get effective settings
        settings = _get_effective_settings(project, tags)

        # determine capture settings
        should_capture_input = capture_input if capture_input is not None else _settings.log_inputs
        should_capture_output = capture_output if capture_output is not None else _settings.log_outputs

        # build opik.track arguments
        track_args: Dict[str, Any] = {
            "type": span_type,
            "capture_input": should_capture_input,
            "capture_output": should_capture_output,
            **opik_kwargs,
        }

        # add project if available
        if settings["project"]:
            track_args["project_name"] = settings["project"]

        if name:
            track_args["name"] = name

        # add tags and metadata
        if settings["tags"] or metadata:
            combined_metadata = {**(metadata or {})}
            if settings["tags"]:
                combined_metadata["tags"] = settings["tags"]
            track_args["metadata"] = combined_metadata

        # apply opik.track decorator
        # Return function with type annotations persisted for static type checking
        return cast(Callable[P, R], opik.track(**track_args)(func))

    return decorator


class Trackable:
    """Base class for objects that support automatic tracking.

    Manages the attributes project, and commit for classes that subclass it.
    All Modaic classes except PrecompiledProgram should inherit from this class.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        commit: Optional[str] = None,
        trace: bool = False,
    ):
        self.project = project
        self.commit = commit
        self.trace = trace

    def set_project(self, project: Optional[str] = None, trace: bool = True):
        """Update the project for this trackable object."""
        self.project = project
        self.trace = trace


MethodDecorator = Callable[
    [Callable[Concatenate[T, P], R]],
    Callable[Concatenate[T, P], R],
]


def track_modaic_obj(func: Callable[Concatenate[T, P], R]) -> Callable[Concatenate[T, P], R]:
    """Method decorator for Trackable objects to automatically track method calls.

    Uses self.project to automatically set project
    for modaic.track, then wraps the function with modaic.track.

    Usage:
        class Retriever(Trackable):
            @track_modaic_obj
            def retrieve(self, query: str):
                ...
    """

    @wraps(func)
    def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
        # self should be a Trackable instance
        # TODO: may want to get rid of this type check for hot paths
        if not isinstance(self, Trackable):
            raise ValueError("@track_modaic_obj can only be used on methods of Trackable subclasses")

        # get project from self
        project = getattr(self, "project", None)

        # check if tracking is enabled both globally and for this object
        if not _settings.tracing or not self.trace:
            # binds the method to self so it can be called with args and kwars, also type cast's it to callable with type vars for static type checking
            bound = cast(Callable[P, R], func.__get__(self, type(self)))
            return bound(*args, **kwargs)

        # create tracking decorator with automatic name generation
        tracker = track(name=f"{self.__class__.__name__}.{func.__name__}", project=project, span_type="general")

        # apply tracking and call method
        # type casts the 'track' decorator static type checking
        tracked_func = cast(MethodDecorator, tracker)(func)
        # binds the method to self so it can be called with args and kwars, also type cast's it to callable with type vars for static type checking
        bound_tracked = cast(Callable[P, R], tracked_func.__get__(self, type(self)))
        return bound_tracked(*args, **kwargs)

    return cast(Callable[Concatenate[T, P], R], wrapper)
