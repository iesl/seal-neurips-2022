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

import allennlp

logger = logging.getLogger(__name__)


class InstanceFields(TypedDict):
    """Contents which form an instance"""

    x: TextField  #:
    labels: MultiLabelField  #: types


@DatasetReader.register("bgc")
class BlurbGenreReader(DatasetReader):
    """
    Multi-label classification `dataset <https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html>`_.

    The output of this DatasetReader follows :class:`MultiInstanceEntityTyping`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        use_transitive_closure: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Arguments:
            tokenizer: The tokenizer to be used.
            token_indexers: The token_indexers to be used--one per embedder. Will usually be only one.
            use_transitive_closure: use types_extra
            **kwargs: Parent class args.
                `Reference <https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/dataset_reader.py>`_

        """
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self._use_transitive_closure = use_transitive_closure

    def example_to_fields(
        self,
        title: str,
        body: str,
        topics: Dict[str, List[str]],
        idx: str,
        meta: Dict = None,
        **kwargs: Any,
    ) -> InstanceFields:
        """Converts a dictionary containing an example datapoint to fields that can be used
        to create an :class:`Instance`. If a meta dictionary is passed, then it also adds raw data in
        the meta dict.

        Args:
            title: Title of the book
            body: Blurb
            topics: Labels/Genres
            idx: unique id
            meta: None
            **kwargs: TODO

        Returns:
            Dictionary of fields with the following entries:
                sentence: contains the body.
                mention: contains the title.

        """

        if meta is None:
            meta = {}

        meta["title"] = title
        meta["body"] = body
        meta["topics"] = topics
        meta["idx"] = idx
        meta["using_tc"] = False

        x = TextField(
            self._tokenizer.tokenize(body),
        )

        labels_: List[str] = []

        for value in topics.values():
            labels_ += (
                [value]  # type:ignore
                if not isinstance(value, list)
                else value
            )
        labels = MultiLabelField(labels_)

        return {
            "x": x,
            "labels": labels,
        }

    def text_to_instance(  # type:ignore
        self,
        title: str,
        body: str,
        topics: Dict[str, List[str]],
        idx: str,
        **kwargs: Any,
    ) -> Instance:
        """Converts contents of a single raw example to :class:`Instance`.

        Args:
            title: Title of the book
            body: Blurb Text
            topics: Labels/Genres of the book
            idx: Identification number
            **kwargs: unused

        Returns:
             :class:`Instance` of data

        """
        meta_dict: Dict = {}
        main_fields = self.example_to_fields(
            title, body, topics, idx, meta=meta_dict
        )

        return Instance(
            {**cast(dict, main_fields), "meta": MetadataField(meta_dict)}
        )

    def _read(self, file_path: str) -> Iterator[Instance]:
        """Reads a datafile to produce instances

        Args:
            file_path: TODO

        Yields:
            data instances

        """

        for file_ in glob.glob(file_path, flags=glob.EXTGLOB):
            # logger.info(f"Reading {file_}")
            with open(file_) as f:
                for line in self.shard_iterable(f):
                    example = json.loads(line)
                    instance = self.text_to_instance(**example)
                    yield instance

    def apply_token_indexers(self, instance: Instance) -> None:
        text_field = cast(TextField, instance.fields["x"])  # no runtime effect
        text_field._token_indexers = self._token_indexers
