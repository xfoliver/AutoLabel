"""PyQt 对话框集合。"""

from dialogs.documentation import open_readme, show_documentation_dialog
from dialogs.post_train import PostTrainDialog
from dialogs.submission_history import (
    RecentSubmissionsCard,
    SubmissionHistoryDialog,
)
from dialogs.templates import (
    ClickableFrame,
    QuickTemplatesDialog,
    SaveTemplateDialog,
)

__all__ = [
    "ClickableFrame",
    "PostTrainDialog",
    "QuickTemplatesDialog",
    "RecentSubmissionsCard",
    "SaveTemplateDialog",
    "SubmissionHistoryDialog",
    "open_readme",
    "show_documentation_dialog",
]
