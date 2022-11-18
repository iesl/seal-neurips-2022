from typing import (
    Dict,
    List,
    Union,
    Any,
    Iterator,
    Tuple,
    cast,
    Optional,
    Iterable,
)
import sys
import itertools
if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict, Literal, overload

import logging
import json
import dill
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    TextField,
    Field,
    ArrayField,
    ListField,
    MetadataField,
    MultiLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers import Token
from .blurb_genre_collection import BlurbGenreReader
import glob

logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    x: TextField  #:
    labels: MultiLabelField  #: types


@DatasetReader.register("nyt")
class NytReader(BlurbGenreReader):
    """
    Reader for the New York times dataset

    """

    def example_to_fields(
        self,
        text: str,
        title: str,
        labels: List[str],
        general_descriptors: List[List[str]],
        label_paths: List[List[str]],
        xml_path: str,
        taxonomy: List[str],
        meta: Dict = None,
        **kwargs: Any,
    ) -> InstanceFields:
        """Converts a dictionary containing an example datapoint to fields that can be used
        to create an :class:`Instance`. If a meta dictionary is passed, then it also adds raw data in
        the meta dict.

        Args:
            text: One line summary of article,
            title: Title of the article
            labels:list of labels,
            general_descriptors: Extra descriptors,
            label_path: List of taxonomies,
            xml_path: path to xml file,
            taxonomy: Taxonomy extracted form xml file
            **kwargs: Any
        Returns:
            Dictionary of fields with the following entries:
                sentence: contains the body.
                mention: contains the title.

        """

        if meta is None:
            meta = {}

        meta["text"] = text
        meta["labels"] = labels
        meta["general_descriptors"] = general_descriptors
        meta["label_path"] = label_paths
        meta["xml_path"] = xml_path
        meta["taxonomy"] = taxonomy

        x = TextField(self._tokenizer.tokenize(text))
        labels = MultiLabelField(labels)

        return {
            "x": x,
            "labels": labels,
        }

    def text_to_instance(  # type:ignore
        self,
        text: str,
        title: str,
        labels: List[str],
        general_descriptors: List[str],
        label_paths: List[List[str]],
        xml_path: str,
        taxonomy: List[str],
        **kwargs: Any
    ) -> Instance:
        """Converts contents of a single raw example to :class:`Instance`.

        Args:
            text: One line summary of article,
            title: Title of the article
            labels:list of labels,
            general_descriptors: Extra descriptors,
            label_path: List of taxonomies,
            xml_path: path to xml file,
            taxonomy: Taxonomy extracted form xml file
            **kwargs: Any

        Returns:
             :class:`Instance` of data

        """
        meta_dict: Dict = {}
        main_fields = self.example_to_fields(
            text, title, labels, general_descriptors, label_paths, xml_path, taxonomy, meta=meta_dict
        )

        return Instance(
            {**cast(dict, main_fields), "meta": MetadataField(meta_dict)}
        )

