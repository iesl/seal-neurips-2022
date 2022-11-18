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
from wcmatch import glob

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
import allennlp


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    x: TextField  #:
    labels: MultiLabelField  #: types


@DatasetReader.register("rcv")
class RCV(BlurbGenreReader):
    def example_to_fields(
        self,
        headline: str,
        text: str,
        labels: List[str],
        idx: str,
        meta: Dict = None,
        **kwargs: Any,
    ) -> InstanceFields:
        """Converts a dictionary containing an example datapoint to fields that can be used
        to create an :class:`Instance`. If a meta dictionary is passed, then it also adds raw data in
        the meta dict.

        Returns:
        Dictionary of fields with the following entries:
            sentence: contains the body.
            mention: contains the title.

        """
        if meta is None:
            meta = {}

        meta["headline"] = headline
        # meta["text"] = text
        meta["labels"] = labels
        meta["idx"] = idx

        x = TextField(self._tokenizer.tokenize(text),)


        return {
            "x": x,
            "labels": MultiLabelField(labels),
        }

    def text_to_instance(  # type:ignore
        self,
        headline: str,
        text: str,
        labels: List[str],
        idx: str,
        **kwargs: Any,
    ) -> Instance:
        """Converts contents of a single raw example to :class:`Instance`.

        Returns:
         :class:`Instance` of data

        """
        meta_dict: Dict = {}
        main_fields = self.example_to_fields(
            headline or "", text or "", labels, idx, meta=meta_dict
        )

        return Instance(
            {**cast(dict, main_fields), "meta": MetadataField(meta_dict)}
        )
