# -*- coding: utf8 -*-

import dataclasses
import json
import logging
import os
import typing as t
from pathlib import Path

import docspec
import typing_extensions as te
from databind.core import DeserializeAs
import yaml

from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
from pydoc_markdown.interfaces import Context, Renderer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CustomizedMarkdownRenderer(MarkdownRenderer):
    """We override some defaults in this subclass."""

    #: Disabled because Docusaurus supports this automatically.
    insert_header_anchors: bool = False

    #: Escape html in docstring, otherwise it could lead to invalid html.
    escape_html_in_docstring: bool = True

    #: Conforms to Docusaurus header format.
    render_module_header_template: str = (
        "---\nsidebar_label: {relative_module_name}\ntitle: {module_name}\n---\n\n"
    )


@dataclasses.dataclass
class NavPage:
    """Navigation page metadata used for frontmatter injection and docs.json.

    Params:
        title: The page title.
        description: Optional short description.
        sidebarTitle: Optional alternative sidebar title.
        icon: Optional icon name.
        iconType: Optional icon type (e.g., regular).
        tag: Optional badge text.
        mode: Optional layout mode (e.g., wide).
        url: Optional external URL.
        keywords: Optional list of keywords.
        openapi: Optional OpenAPI endpoint string.
        path: Optional docs-relative path without extension (e.g., reference/pkg/mod).
    """

    title: t.Optional[str] = None
    description: t.Optional[str] = None
    sidebarTitle: t.Optional[str] = None
    icon: t.Optional[str] = None
    iconType: t.Optional[str] = None
    tag: t.Optional[str] = None
    mode: t.Optional[str] = None
    url: t.Optional[str] = None
    keywords: t.Optional[t.List[str]] = None
    openapi: t.Optional[str] = None
    path: t.Optional[str] = None


@dataclasses.dataclass
class NavGroup:
    """Group of pages within a tab.

    Params:
        group: Group display name.
        pages: The list of pages under this group. Can contain page strings,
               page objects with extra frontmatter, or nested groups.
    """

    group: str
    pages: t.List["NavEntry"] = dataclasses.field(default_factory=list)


NavEntry = t.Union["NavGroup", NavPage, str]


@dataclasses.dataclass
class NavTab:
    """Navigation tab configuration.

    Params:
        tab: The tab display name.
        pages: The list of pages belonging to this tab.
        icon: Optional tab-level icon.
        groups: Optional list of groups within the tab.
        anchors: Optional anchors configuration for the tab.
    """

    tab: str
    pages: t.List[NavEntry] = dataclasses.field(default_factory=list)
    icon: t.Optional[str] = None
    groups: t.Optional[t.List["NavGroup"]] = None
    anchors: t.Optional[t.Any] = None


@dataclasses.dataclass
class Navigation:
    """Root navigation configuration consumed from the YAML config.

    Params:
        tabs: The list of tabs in the navigation.
        menu: Optional menu configuration displayed in navigation.
        dropdowns: Optional dropdowns configuration displayed in navigation.
    """

    tabs: t.List[NavTab] = dataclasses.field(default_factory=list)
    menu: t.Optional[t.List[t.Dict[str, t.Any]]] = None
    dropdowns: t.Optional[t.List[t.Dict[str, t.Any]]] = None


@dataclasses.dataclass
class MintlifyRenderer(Renderer):
    """Render Markdown and Mintlify navigation.

    Produces Markdown files and a `docs.json` for use in a Mintlify site. Files are rendered
    under `docs_base_path/relative_output_path` and navigation/frontmatter are driven by the
    optional `navigation` and `docs_json` configuration.

    Params:
        markdown: Markdown renderer configuration.
        docs_base_path: Root docs directory.
        relative_output_path: Subfolder for API reference output.
        relative_sidebar_path: Output JSON file for the sidebar tree (legacy artifact).
        sidebar_top_level_label: Top-level label for the generated sidebar JSON.
        sidebar_top_level_module_label: Optional override label for the top-level module.
        navigation: Optional navigation configuration to build tabs/pages and frontmatter.
        docs_json: Optional global docs.json configuration merged with generated navigation.
    """

    #: The #MarkdownRenderer configuration.
    markdown: te.Annotated[
        MarkdownRenderer, DeserializeAs(CustomizedMarkdownRenderer)
    ] = dataclasses.field(default_factory=CustomizedMarkdownRenderer)

    #: The path where the docusaurus docs content is. Defaults "docs" folder.
    docs_base_path: str = "docs"

    #: The output path inside the docs_base_path folder, used to output the
    #: module reference.
    relative_output_path: str = "reference"

    #: The sidebar path inside the docs_base_path folder, used to output the
    #: sidebar for the module reference.
    relative_sidebar_path: str = "sidebar.json"

    #: The top-level label in the sidebar. Default to 'Reference'. Can be set to null to
    #: remove the sidebar top-level all together. This option assumes that there is only one top-level module.
    sidebar_top_level_label: t.Optional[str] = "Reference"

    #: The top-level module label in the sidebar. Default to null, meaning that the actual
    #: module name will be used. This option assumes that there is only one top-level module.
    sidebar_top_level_module_label: t.Optional[str] = None

    #: Optional navigation configuration to generate Mintlify tabs/pages.
    navigation: t.Optional[Navigation] = None

    # Global docs.json configuration fields (top-level convenience fields)
    theme: t.Optional[str] = None
    name: t.Optional[str] = None
    colors: t.Optional[t.Dict[str, t.Any]] = None
    description: t.Optional[str] = None
    logo: t.Optional[t.Union[str, t.Dict[str, t.Any]]] = None
    favicon: t.Optional[t.Union[str, t.Dict[str, t.Any]]] = None
    thumbnails: t.Optional[t.Dict[str, t.Any]] = None
    styling: t.Optional[t.Dict[str, t.Any]] = None
    icons: t.Optional[t.Dict[str, t.Any]] = None
    fonts: t.Optional[t.Dict[str, t.Any]] = None
    appearance: t.Optional[t.Dict[str, t.Any]] = None
    background: t.Optional[t.Dict[str, t.Any]] = None
    navbar: t.Optional[t.Dict[str, t.Any]] = None
    links: t.Optional[t.List[t.Dict[str, t.Any]]] = None
    primary: t.Optional[t.Dict[str, t.Any]] = None
    footer: t.Optional[t.Dict[str, t.Any]] = None
    banner: t.Optional[t.Dict[str, t.Any]] = None
    redirects: t.Optional[t.List[t.Dict[str, t.Any]]] = None
    contextual: t.Optional[t.Dict[str, t.Any]] = None
    api: t.Optional[t.Dict[str, t.Any]] = None
    seo: t.Optional[t.Dict[str, t.Any]] = None
    search: t.Optional[t.Dict[str, t.Any]] = None
    integrations: t.Optional[t.Dict[str, t.Any]] = None
    errors: t.Optional[t.Dict[str, t.Any]] = None

    def init(self, context: Context) -> None:
        """Initialize the underlying Markdown renderer.

        Params:
            context: Pydoc-Markdown context.
        """
        self.markdown.init(context)

    def render(self, modules: t.List[docspec.Module]) -> None:
        """Render modules to MDX, inject page frontmatter, and write docs.json.

        Params:
            modules: The list of modules to render.
        """
        module_tree: t.Dict[str, t.Any] = {"children": {}, "edges": []}
        output_path = Path(self.docs_base_path) / self.relative_output_path

        # Build lookup map from configured navigation pages by their path, traversing
        # nested groups and allowing simple string page entries.
        page_by_path: t.Dict[str, NavPage] = {}

        def collect_pages(entries: t.Iterable[NavEntry]) -> None:
            for entry in entries:
                if isinstance(entry, str):
                    # Simple path-only entry
                    path_value = entry
                    if path_value:
                        page_by_path.setdefault(path_value, NavPage(path=path_value))
                elif isinstance(entry, NavPage):
                    if entry.path:
                        page_by_path[entry.path] = entry
                else:
                    # NavGroup – recurse
                    collect_pages(entry.pages)

        if self.navigation:
            for tab in self.navigation.tabs:
                collect_pages(tab.pages)
                if tab.groups:
                    for group in tab.groups:
                        collect_pages(group.pages)

        # Disable default module header to avoid duplicated frontmatter.
        self.markdown.render_module_header = False

        for module in modules:
            filepath = output_path

            module_parts = module.name.split(".")
            if module.location.filename.endswith("__init__.py"):
                module_parts.append("__init__")

            relative_module_tree = module_tree
            intermediary_module = []

            for module_part in module_parts[:-1]:
                # update the module tree
                intermediary_module.append(module_part)
                intermediary_module_name = ".".join(intermediary_module)
                relative_module_tree["children"].setdefault(
                    intermediary_module_name, {"children": {}, "edges": []}
                )
                relative_module_tree = relative_module_tree["children"][
                    intermediary_module_name
                ]

                # descend to the file
                filepath = filepath / module_part

            # create intermediary missing directories and get the full path
            filepath.mkdir(parents=True, exist_ok=True)
            filepath = filepath / f"{module_parts[-1]}.mdx"

            # Compute docs-relative path without extension, used to match a page.
            rel_without_ext = os.path.splitext(
                str(filepath.relative_to(self.docs_base_path))
            )[0]

            with filepath.open("w", encoding=self.markdown.encoding) as fp:
                logger.info("Render file %s", filepath)

                # Inject frontmatter if a page is configured, otherwise provide a minimal title.
                frontmatter: t.Dict[str, t.Any] = {}
                page = page_by_path.get(rel_without_ext)
                if page:
                    # Copy all provided page fields except path
                    for key in [
                        "title",
                        "description",
                        "sidebarTitle",
                        "icon",
                        "iconType",
                        "tag",
                        "mode",
                        "url",
                        "keywords",
                        "openapi",
                    ]:
                        value = getattr(page, key)
                        if value is not None:
                            frontmatter[key] = value
                else:
                    frontmatter["title"] = module.name

                # Write YAML frontmatter header.
                fp.write("---\n")
                fp.write(yaml.safe_dump(frontmatter, sort_keys=False))
                fp.write("---\n\n")

                # Render the API content below the frontmatter.
                self.markdown.render_single_page(fp, [module])

            # only update the relative module tree if the file is not empty
            relative_module_tree["edges"].append(
                os.path.splitext(str(filepath.relative_to(self.docs_base_path)))[0]
            )

        self._render_side_bar_config(module_tree)

        # Render docs.json with merged global configuration and navigation tabs/pages.
        self._render_docs_json()

    def _render_side_bar_config(self, module_tree: t.Dict[t.Text, t.Any]) -> None:
        """Render legacy sidebar JSON for reference browsing.

        Params:
            module_tree: Generated module tree.
        """
        sidebar: t.Dict[str, t.Any] = {
            "type": "category",
            "label": self.sidebar_top_level_label,
        }
        self._build_sidebar_tree(sidebar, module_tree)

        if sidebar.get("items"):
            if self.sidebar_top_level_module_label:
                sidebar["items"][0]["label"] = self.sidebar_top_level_module_label

            if not self.sidebar_top_level_label:
                # it needs to be a dictionary, not a list; this assumes that
                # there is only one top-level module
                sidebar = sidebar["items"][0]

        sidebar_path = (
            Path(self.docs_base_path)
            / self.relative_output_path
            / self.relative_sidebar_path
        )
        # with sidebar_path.open("w") as handle:
        #     logger.info("Render file %s", sidebar_path)
        #     json.dump(sidebar, handle, indent=2, sort_keys=True)

    def _build_sidebar_tree(
        self, sidebar: t.Dict[t.Text, t.Any], module_tree: t.Dict[t.Text, t.Any]
    ) -> None:
        """Recursively build the sidebar tree.

        Params:
            sidebar: Sidebar dictionary to mutate.
            module_tree: Generated module tree.
        """
        sidebar["items"] = module_tree.get("edges", [])
        if os.name == "nt":
            # Make generated configuration more portable across operating systems (see #129).
            sidebar["items"] = [x.replace("\\", "/") for x in sidebar["items"]]
        for child_name, child_tree in module_tree.get("children", {}).items():
            child = {
                "type": "category",
                "label": child_name,
            }
            self._build_sidebar_tree(child, child_tree)
            sidebar["items"].append(child)

        def _sort_items(
            item: t.Union[t.Text, t.Dict[t.Text, t.Any]],
        ) -> t.Tuple[int, t.Text]:
            """Sort sidebar items. Order follows:
            1. modules containing items come first
            2. alphanumeric order is applied
            """
            is_edge = int(isinstance(item, str))
            label = item if is_edge else item.get("label")  # type: ignore
            return is_edge, str(label)

        sidebar["items"] = sorted(sidebar["items"], key=_sort_items)

    def _render_docs_json(self) -> None:
        """Generate `docs.json` using the provided global configuration and navigation.

        Writes the file to `<docs_base_path>/docs.json`.
        """
        docs_path = Path(self.docs_base_path) / "docs.json"

        # Collect top-level global configuration from renderer fields.
        config: t.Dict[str, t.Any] = {}

        def set_if_present(key: str, value: t.Any) -> None:
            if value is not None:
                config[key] = value

        set_if_present("theme", self.theme)
        set_if_present("name", self.name)
        set_if_present("colors", self.colors)
        set_if_present("description", self.description)
        set_if_present("logo", self.logo)
        set_if_present("favicon", self.favicon)
        set_if_present("thumbnails", self.thumbnails)
        set_if_present("styling", self.styling)
        set_if_present("icons", self.icons)
        set_if_present("fonts", self.fonts)
        set_if_present("appearance", self.appearance)
        set_if_present("background", self.background)
        set_if_present("navbar", self.navbar)
        set_if_present("links", self.links)
        set_if_present("primary", self.primary)
        set_if_present("footer", self.footer)
        set_if_present("banner", self.banner)
        set_if_present("redirects", self.redirects)
        set_if_present("contextual", self.contextual)
        set_if_present("api", self.api)
        set_if_present("seo", self.seo)
        set_if_present("search", self.search)
        set_if_present("integrations", self.integrations)
        set_if_present("errors", self.errors)

        # Build navigation from tabs/pages if provided.
        if self.navigation:
            tabs_out: t.List[t.Dict[str, t.Any]] = []
            for tab in self.navigation.tabs:
                tab_obj: t.Dict[str, t.Any] = {"tab": tab.tab}
                if tab.icon:
                    tab_obj["icon"] = tab.icon

                def serialize_entries(
                    entries: t.Iterable[NavEntry],
                ) -> t.List[t.Union[str, t.Dict[str, t.Any]]]:
                    serialized: t.List[t.Union[str, t.Dict[str, t.Any]]] = []
                    for entry in entries:
                        if isinstance(entry, str):
                            serialized.append(entry)
                        elif isinstance(entry, NavPage):
                            if entry.path:
                                serialized.append(entry.path)
                        else:
                            # NavGroup: recurse
                            child_serialized = serialize_entries(entry.pages)
                            if child_serialized:
                                serialized.append(
                                    {
                                        "group": entry.group,
                                        "pages": child_serialized,
                                    }
                                )
                    return serialized

                serialized_pages = serialize_entries(tab.pages)
                if serialized_pages:
                    tab_obj["pages"] = serialized_pages

                # groups
                if tab.groups:
                    groups_out: t.List[t.Dict[str, t.Any]] = []
                    for group in tab.groups:
                        child_serialized = serialize_entries(group.pages)
                        groups_out.append(
                            {
                                "group": group.group,
                                "pages": child_serialized,
                            }
                        )
                    if groups_out:
                        tab_obj["groups"] = groups_out
                # anchors
                if tab.anchors is not None:
                    tab_obj["anchors"] = tab.anchors
                tabs_out.append(tab_obj)

            # Merge or set navigation structure.
            navigation_obj: t.Dict[str, t.Any] = config.get("navigation", {})
            navigation_obj["tabs"] = tabs_out
            # root-level menu and dropdowns
            if self.navigation.menu is not None:
                navigation_obj["menu"] = self.navigation.menu
            if self.navigation.dropdowns is not None:
                navigation_obj["dropdowns"] = self.navigation.dropdowns
            config["navigation"] = navigation_obj

        # Default schema if not provided.
        config.setdefault("$schema", "https://mintlify.com/docs.json")

        with docs_path.open("w", encoding="utf-8") as handle:
            logger.info("Render file %s", docs_path)
            json.dump(config, handle, indent=2, sort_keys=False)
