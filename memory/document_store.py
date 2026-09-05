"""Compatibility exports for the document-storage module layout."""

from .storage import BaseDocumentStore, SQLiteDocumentStore

__all__ = ["BaseDocumentStore", "SQLiteDocumentStore"]
