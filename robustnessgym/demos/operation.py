# flake8: noqa
import glob
import inspect
from pathlib import Path

import numpy as np
import pyarrow
import streamlit as st
from meerkat.columns.prediction_column import ClassificationOutputColumn

import robustnessgym as rg
from robustnessgym import DataPanel


# @st.cache(hash_funcs={pyarrow.lib.Buffer: lambda x: 0})
def load_boolq():
    return DataPanel.load_huggingface(
        "boolq",
        split="validation",
    )


# @st.cache#(hash_funcs={pyarrow.lib.Buffer: lambda x: 0})
def load_boolq_small():
    return DataPanel.load_huggingface("boolq", split="validation[:2]")


def load_datapanel(dataset="boolq", size="small"):
    if dataset == "boolq":
        if size == "small":
            return load_boolq_small()
        else:
            return load_boolq()


def format_code(code=None, output=None, outputs=None, columns=(0.45, 0.1, 0.45)):
    with st.beta_container():
        code_col, _, stdout_col = st.beta_columns(columns)
        with code_col:
            st.code(code)
        with stdout_col:
            if output is not None:
                st.write(output)
            else:
                for output in outputs:
                    st.write(output)
    st.write("<br>", unsafe_allow_html=True)


def datapanel_page(dataset="boolq", size="small"):
    st.write("# DataPanels: Get your data into Robustness Gym")
    st.write(
        """
Load data into Robustness Gym using the `DataPanel`. For detailed info about 
the `DataPanel`, check out [Mosaic](https://github.com/robustness-gym/meerkat).
        """
    )

    dp = display_dp(dataset, size)

    st.write(
        """
The `DataPanel` organizes data into columns, where columns have heterogenous types.
For example, in the `DataPanel` we loaded, some columns are of type `ListColumn`,
while one is of type `NumpyArrayColumn`. 
        """
    )

    datapanel_sheet(dp)


def datapanel_sheet(dp):
    with st.beta_container():
        code_col, _, stdout_col = st.beta_columns([0.30, 0.05, 0.65])
        with code_col:
            st.write("## Code Snippet")
        with stdout_col:
            st.write("## Output")

    format_code("dp", dp.streamlit(), columns=[0.30, 0.05, 0.65])
    format_code(
        """
dp['passage']
type(dp['passage'])
        """,
        outputs=(dp["passage"].streamlit(), str(type(dp["passage"]))),
        columns=[0.30, 0.05, 0.65],
    )
    format_code(
        """
dp['answer']
type(dp['answer'])
        """,
        outputs=(dp["answer"].streamlit(), str(type(dp["answer"]))),
        columns=[0.30, 0.05, 0.65],
    )
    format_code(
        """
# NumpyArrayColumn works like np.ndarray!
dp['answer'].mean()
dp['answer'].reshape(-1, 1, 1)
        """
        "",
        outputs=(dp["answer"].mean(), dp["answer"].reshape(-1, 1, 1).streamlit()),
        columns=[0.30, 0.05, 0.65],
    )
    format_code("dp[0]", dp[0], columns=[0.30, 0.05, 0.65])
    format_code("dp[:1]", dp[:1].streamlit(), columns=[0.30, 0.05, 0.65])
    format_code("dp.identifier", dp.identifier, columns=[0.30, 0.05, 0.65])
    format_code("dp.columns", dp.columns, columns=[0.30, 0.05, 0.65])
    format_code(
        "dp.map(lambda x: len(x['passage']))",
        dp.map(lambda x: len(x["passage"])).streamlit(),
        columns=[0.30, 0.05, 0.65],
    )
    format_code(
        """
dp.map(lambda x: {
    'passage_len': len(x['passage']),  
    'question_len': len(x['question']),
})
        """,
        dp.map(
            lambda x: {
                "passage_len": len(x["passage"]),
                "question_len": len(x["question"]),
            }
        ).streamlit(),
        columns=[0.30, 0.05, 0.65],
    )
    format_code(
        "dp.filter(lambda x: len(x['passage']) > 600)",
        dp.filter(lambda x: len(x["passage"]) > 600).streamlit(),
        columns=[0.30, 0.05, 0.65],
    )
    format_code(
        "dp.update(lambda x: {'passage_len': len(x['passage'])})",
        dp.update(lambda x: {"passage_len": len(x["passage"])}).streamlit(),
        columns=[0.30, 0.05, 0.65],
    )
    format_code(
        """
dp.update(lambda x: {
    'passage_len': len(x['passage']),  
    'question_len': len(x['question']),
})
        """,
        dp.update(
            lambda x: {
                "passage_len": len(x["passage"]),
                "question_len": len(x["question"]),
            }
        ).streamlit(),
        columns=[0.30, 0.05, 0.65],
    )
    format_code("dp.to_pandas()", dp.to_pandas(), columns=[0.30, 0.05, 0.65])

    st.write("------")


def display_dp(dataset="boolq", size="small"):
    st.subheader("Load Data")
    code_col, stdout_col = st.beta_columns(2)
    with code_col:
        if dataset == "boolq":
            if size == "small":
                dp = load_boolq_small()
                st.code(
                    """
# use the DataPanel to get your data into RG
from robustnessgym import DataPanel  
# any Huggingface dataset
dp = DataPanel.load_huggingface('boolq', split='validation[:2]')  
                    """
                )
            else:
                dp = load_boolq()
                st.code(
                    """
# use the DataPanel to get your data into RG
from robustnessgym import DataPanel
# any Huggingface dataset
dp = DataPanel.load_huggingface('boolq', split='validation')
                    """
                )

    return dp


def operation_page():
    st.write("# Operations: Run common processing workflows in Robustness Gym")
    st.write(
        """
An `Operation` in Robustness Gym is used to write and run common workflows 
e.g. tokenizing text, computing embeddings, tagging text, etc. 
Run an Operation on a `DataPanel` to
add new columns to the `DataPanel` that can be used for analysis. 

The main thing to remember: an `Operation` adds new columns to your `DataPanel`.
        """
    )

    dp = display_dp()

    with st.beta_expander("Run the spaCy pipeline with `SpacyOp`"):
        spacy_example(dp)

    with st.beta_expander("Run the Stanza pipeline with `StanzaOp`"):
        stanza_example(dp)

    with st.beta_expander("Lazily run the TextBlob pipeline with `LazyTextBlobOp`"):
        lazy_textblob_example(dp)

    with st.beta_expander("Write a simple `Operation` to capitalize text"):
        custom_operation_example_1(dp)

    with st.beta_expander(
        "Write an `Operation` that adds "
        "multiple columns to capitalize and upper-case text"
    ):
        custom_operation_example_2(dp)


def spacy_example(dp):
    columns = [0.45, 0.05, 0.50]

    with st.beta_container():
        st.header("Run Operation: `SpacyOp`")
        st.write(
            """
spaCy is a popular text processing library that provides tokenization, tagging 
and other capabilities. 
            """
        )

        from robustnessgym import lookup
        from robustnessgym.ops import SpacyOp

        # Run the Spacy pipeline on the 'question' column of the dataset
        spacy = SpacyOp()
        dp = spacy(dp=dp, columns=["passage"])
        # adds a new column that is auto-named
        # "SpacyOp(lang=en_core_web_sm, neuralcoref=False, columns=['passage'])"

        format_code(
            """
from robustnessgym import lookup
from robustnessgym.ops import SpacyOp

# Run the Spacy pipeline on the 'question' column of the dataset
spacy = SpacyOp()
dp = spacy(dp=dp, columns=['passage'])
# adds a new column that is auto-named
# "SpacyOp(lang=en_core_web_sm, neuralcoref=False, columns=['passage'])"
            """,
            dp.streamlit(),
            columns=columns,
        )

    st.write("------")

    with st.beta_container():
        spacy_column = lookup(dp, spacy, ["passage"])
        format_code(
            """
lookup(dp, spacy, ['passage'])
            """,
            spacy_column._repr_pandas_(),
            columns=columns,
        )

    with st.beta_container():
        st.subheader("Columns contain Cells")
        cell = spacy_column[1]
        format_code(
            """
cell = spacy_column[1]
cell
            """,
            cell,
            columns=columns,
        )

        format_code(
            """
list(cell)
            """,
            list(cell),
            columns=columns,
        )
        format_code(
            """
cell.ents
            """,
            cell.ents,
            columns=columns,
        )


def stanza_example(dp):
    columns = [0.45, 0.05, 0.50]
    st.header("Run Stanza Workflow")
    from robustnessgym import lookup
    from robustnessgym.ops import StanzaOp

    # Run the Stanza pipeline on the 'question' column of the dataset
    stanza = StanzaOp()
    dp = stanza(dp=dp, columns=["question"])
    # adds a new column that is auto-named "StanzaOp(columns=['question'])"

    # Grab the Stanza column from the DataPanel using the lookup
    stanza_column = lookup(dp, stanza, ["question"])
    format_code(
        """
from robustnessgym import lookup
from robustnessgym.ops import StanzaOp

# Run the Stanza pipeline on the 'question' column of the dataset
stanza = StanzaOp()
dp = stanza(dp=dp, columns=['question'])
# adds a new column that is auto-named "StanzaOp(columns=['question'])"

# Grab the Stanza column from the DataPanel using the lookup
stanza_column = lookup(dp, stanza, ['question'])
        """,
        stanza_column._repr_pandas_(),
        columns=columns,
    )
    st.subheader("Columns contain Cells")
    format_code(
        """ 
cell = stanza_column[0]
cell
        """,
        stanza_column[0],
        columns=columns,
    )
    st.subheader("Cells can be treated like stanza objects")
    format_code(
        """ 
cell = stanza_column[0]
cell.text
        """,
        stanza_column[0].text,
        columns=columns,
    )
    format_code(
        """ 
cell = stanza_column[0]
cell.entities
        """,
        stanza_column[0].entities,
        columns=columns,
    )


def lazy_textblob_example(dp):
    columns = [0.45, 0.05, 0.50]
    st.header("Run TextBlob Workflow Lazily")

    def fn():
        nonlocal dp
        from robustnessgym import lookup
        from robustnessgym.ops import LazyTextBlobOp

        # Run the TextBlob pipeline on the 'passage' column of the dataset
        textblob = LazyTextBlobOp()
        dp = textblob(dp=dp, columns=["question"])
        # adds a new column that is auto-named "LazyTextBlobOp(columns=['question'])"

        # Grab the TextBlob column from the DataPanel using the lookup
        textblob_column = lookup(dp, textblob, ["question"])
        return textblob_column

    fn_source = inspect.getsource(fn)
    fn_source = "\n".join([line[4:] for line in fn_source.split("\n")[2:]])
    textblob_column = fn()
    format_code(fn_source, textblob_column._repr_pandas_(), columns=columns)

    st.subheader("Columns contain cells")
    cell = textblob_column[1]
    format_code(
        """
cell = textblob_column[1]
cell
        """,
        textblob_column[1],
        columns=columns,
    )
    format_code(
        """
cell.sentences
        """,
        cell.sentences,
        columns=columns,
    )


def custom_operation_example_1(dp):
    columns = [0.45, 0.05, 0.50]
    st.header("Run Custom Operation")

    def fn():
        from robustnessgym import Id, Operation

        # A function that capitalizes text
        def capitalize(batch, columns):
            return [text.capitalize() for text in batch[columns[0]]]

        # Wrap in an Operation: `process_batch_fn` accepts functions that have
        # exactly 2 arguments: batch and columns, and returns a tuple of outputs
        op = Operation(
            identifier=Id("CapitalizeOp"),
            process_batch_fn=capitalize,
        )
        return op

    fn_source = inspect.getsource(fn)
    fn_source = "\n".join([line[4:] for line in fn_source.split("\n")[1:]])
    op = fn()
    format_code(fn_source, op, columns=columns)

    def fn():
        nonlocal dp
        from robustnessgym import lookup

        dp = op(dp=dp, columns=["question"])

        # Look it up when you need it
        capitalized_text = lookup(dp, op, ["question"])
        return capitalized_text

    fn_source = inspect.getsource(fn)
    fn_source = "\n".join([line[4:] for line in fn_source.split("\n")[1:]])
    capitalized_text = fn()
    format_code(fn_source, capitalized_text._repr_pandas_(), columns=columns)

    format_code("capitalized_text[0]", capitalized_text[0], columns=columns)


def custom_operation_example_2(dp):
    columns = [0.45, 0.05, 0.50]
    st.header("Run Custom Operation")

    def fn():
        from robustnessgym import Id, Operation

        # A function that capitalizes and upper-cases text: this will
        # be used to add two columns to the DataPanel
        def capitalize_and_upper(batch, columns):
            return [text.capitalize() for text in batch[columns[0]]], [
                text.upper() for text in batch[columns[0]]
            ]

        # Wrap in an Operation: `process_batch_fn` accepts functions that have
        # exactly 2 arguments: batch and columns, and returns a tuple of outputs
        op = Operation(
            identifier=Id("ProcessingOp"),
            output_names=["capitalize", "upper"],
            process_batch_fn=capitalize_and_upper,
        )
        return op

    fn_source = inspect.getsource(fn)
    fn_source = "\n".join([line[4:] for line in fn_source.split("\n")[1:]])
    op = fn()
    format_code(fn_source, op, columns=columns)

    def fn():
        nonlocal dp
        # Apply to a DataPanel
        dp = op(dp=dp, columns=["question"])
        return dp

    fn_source = inspect.getsource(fn)
    fn_source = "\n".join([line[4:] for line in fn_source.split("\n")[2:]])
    new_dp = fn()
    format_code(fn_source, new_dp._repr_pandas_(), columns=columns)

    def fn():
        nonlocal dp, op
        from robustnessgym import lookup

        capitalized_text = lookup(dp, op, ["question"], "capitalize")
        return capitalized_text

    fn_source = inspect.getsource(fn)
    fn_source = "\n".join([line[4:] for line in fn_source.split("\n")[2:]])
    capitalized_text = fn()
    format_code(fn_source, capitalized_text._repr_pandas_(), columns=columns)

    def fn():
        nonlocal dp, op
        from robustnessgym import lookup

        upper_text = lookup(dp, op, ["question"], "upper")
        return upper_text

    fn_source = inspect.getsource(fn)
    fn_source = "\n".join([line[4:] for line in fn_source.split("\n")[2:]])
    upper_text = fn()
    format_code(fn_source, upper_text._repr_pandas_(), columns=columns)


def subpopulation_example_1():

    dp = display_dp(size="all")

    col1, col2 = st.beta_columns(2)

    with col2:
        with st.echo():
            from robustnessgym import DataPanel, ScoreSubpopulation, lookup
            from robustnessgym.ops import SpacyOp

            def length(batch: DataPanel, columns: list):
                try:
                    # Take advantage of previously stored Spacy information
                    return [len(doc) for doc in lookup(batch, SpacyOp, columns)]
                except AttributeError:
                    # If unavailable, fall back to splitting text
                    return [len(text.split()) for text in batch[columns[0]]]

            # Create a subpopulation that buckets examples based on length
            length_sp = ScoreSubpopulation(
                intervals=[(0, 10), (10, 20)],
                score_fn=length,
            )

            slices, membership = length_sp(dp=dp, columns=["passage"])
            # `slices` is a list of 2 DataPanels
            # `membership` is an np.ndarray of shape (n x 2)
            st.markdown("`slices`")
            st.write(slices)
            st.markdown("`membership`")
            st.write(membership)

        with st.echo():
            # Create a subpopulation that buckets examples based on length
            length_sp = ScoreSubpopulation(
                # mixture of percentile intervals and raw intervals
                intervals=[("0%", "10%"), ("10%", "20%"), (10, 20)],
                score_fn=length,
            )

            slices, membership = length_sp(dp=dp, columns=["passage"])
            st.write("`slices` sizes:", [len(sl) for sl in slices])
            st.write("`membership` shape:", membership.shape)

        st.write("### Updated intervals")
        with st.echo():
            # The percentile intervals are updated to
            # raw values after running the subpopulation
            st.write(length_sp.intervals)

        st.write("### Examples and average passage length for each slice")
        with st.echo():
            for sl in slices:
                # The slice identifier
                st.write(str(sl.identifier))
                # Look at the first 3 examples from the slice
                st.write(sl.head(3)._repr_pandas_())
                # Calculate the average length of the passages in the slice
                st.write(
                    sl.map(lambda x: length(x, ["passage"]), is_batched_fn=True).mean()
                )


EXAMPLE_DATASETS = [
    (
        "snli",
        [
            "validation",  # "test"
        ],
    ),
    ("hans", ["validation"]),
    # ("anli",
    #  ["dev_r1", "test_r1",
    #   "dev_r2", "test_r2",
    #   "dev_r3", "test_r3"
    #   ]),
    (
        "glue/mnli",
        [
            "validation_matched",  # "validation_mismatched", "test_matched", "test_mismatched"
        ],
    ),
]
EXAMPLE_DATASETS_TO_SPLITS = {x[0]: x[1] for x in EXAMPLE_DATASETS}

EXAMPLE_MODELS = [
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli",
    "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli",
    "textattack/bert-base-uncased-snli",
    "textattack/distilbert-base-cased-snli",
    "textattack/bert-base-uncased-MNLI",
    "textattack/albert-base-v2-snli",
    "facebook/bart-large-mnli",
    "roberta-large-mnli",
    "microsoft/deberta-large-mnli",
    "microsoft/deberta-v2-xxlarge-mnli",
    "typeform/mobilebert-uncased-mnli",
    "huggingface/distilbert-base-uncased-finetuned-mnli",
    "prajjwal1/albert-base-v1-mnli",
    "prajjwal1/bert-tiny-mnli",
    "cross-encoder/nli-deberta-base",
    "squeezebert/squeezebert-mnli",
    "squeezebert/squeezebert-mnli-headless",
]


def select_dataset():
    with st.beta_container():
        col1, col2 = st.beta_columns(2)
        with col1:
            dataset = st.radio(
                "Dataset",
                [t[0] for t in EXAMPLE_DATASETS],
            )
        with col2:
            split = st.radio(
                "Split",
                EXAMPLE_DATASETS_TO_SPLITS[dataset],
            )

    return load_dataset(dataset, split)


# @st.cache(allow_output_mutation=True)
def load_dataset(dataset, split):
    dp = DataPanel.load_huggingface(*dataset.split("/"), split=split)
    return dataset.replace("/", "_"), split, dp


# @st.cache(allow_output_mutation=True)
def load_predictions(dataset, split, dp, dir="nli-preds"):
    filepaths = [Path(path) for path in glob.glob(f"{dir}/*")]

    filepaths = [
        path for path in filepaths if path.name.startswith(f"{dataset}-{split}")
    ]

    model_info = {
        path.name.replace(f"{dataset}-{split}-", ""): ClassificationOutputColumn.read(
            str(path)
        )
        for path in filepaths
    }

    # for model, preds in model_info.items():
    #     dp.add_column(model, preds, overwrite=True)

    for model, preds in model_info.items():
        if (
            model.startswith("huggingface")
            or model.startswith("textattack-bert-base")
            or model.startswith("prajjwal")
            or model.startswith("squeezebert-squeezebert")
            or model.startswith("facebook")
        ):
            preds = preds.preds().map(lambda x: {0: 2, 1: 0, 2: 1}[x.item()])
        else:
            preds = preds.preds().map(lambda x: {0: 0, 1: 1, 2: 2}[x.item()])
        dp.add_column(model, preds, overwrite=True)

    return model_info, filepaths, dp


def run_demo():
    # Pick a dataset
    dataset, split, dp = select_dataset()

    st.header(dataset)
    st.subheader(split)
    # st.write(dp.streamlit())

    # Load predictions
    model_info, filepaths, dp = load_predictions(dataset, split, dp)
    # st.write(dp.streamlit())

    # # Pick models
    # models = st.multiselect("Model(s)",
    #                         [model for model in EXAMPLE_MODELS
    #                          if model.replace("/", "-") in model_info])

    with st.beta_container():
        col1, col2 = st.beta_columns(2)
        with col1:
            model_1 = st.selectbox(
                "Model 1",
                [
                    model
                    for model in EXAMPLE_MODELS
                    if model.replace("/", "-") in model_info
                ],
            )
        with col2:
            model_2 = st.selectbox(
                "Model 2",
                [
                    model
                    for model in EXAMPLE_MODELS
                    if model.replace("/", "-") in model_info
                ],
                index=1,
            )

    # if not models:
    #     st.stop()
    from functools import partial

    def accuracy(x, model):
        return (x[model] == x["label"]).mean()

    db = rg.DevBench()
    db.add_aggregators(
        {
            model: {
                "accuracy": partial(accuracy, model=model),
            }
            for model in model_info
        }
    )

    # st.multiselect(
    #     "Slice(s)",
    #     [rg.LexicalOverlapSubpopulation([('0%', '10%'), ('90%', '100%')])]
    # )

    intervals = [("0%", "10%"), ("20%", "80%"), ("90%", "100%")]
    sps = [rg.HasNegation(), rg.NumTokensSubpopulation(intervals=intervals),] + [
        rg.HansAdverbs(),
        rg.HansConjs(),
        rg.HansAdjectives(),
        rg.HansAdjectivesCompNonEnt(),
        rg.HansAdjectivesCompEnt(),
        rg.HansCalledObjects(),
        rg.HansAdvsEntailed(),
        rg.HansAdvsNonEntailed(),
        rg.HasTemporalPreposition(),
        rg.HasComparison(),
        rg.HasQuantifier(),
        rg.HasPosessivePreposition(),
        rg.HansSingularNouns(),
        rg.HasDefiniteArticle(),
        rg.HasIndefiniteArticle(),
    ]
    # dual_sps = [
    #     rg.LexicalOverlapSubpopulation(intervals),
    # ]

    slices = st.multiselect("Slice(s)", sps, format_func=lambda x: x.__class__.__name__)

    generate_report = True  # st.button("Generate Report")

    if generate_report:
        db.add_slices(dp)

        for sl in slices:
            db(sl, dp, ["premise"])
            db(sl, dp, ["hypothesis"])

        db(
            rg.LexicalOverlapSubpopulation(intervals),
            dp,
            ["premise", "hypothesis"],
        )
        # st.write(db.create_report([model.replace("/", "-") for model in models]).figure())
        # for model in models:

        report = db.create_report([model_1.replace("/", "-")])
        report.update_config(**dict(color_scheme=["#3499ec", "#ec34c1", "#9cec34"]))
        st.write(report.figure())
        st.write(db.create_report([model_2.replace("/", "-")]).figure())

    # for model in models:
    #     st.write(model_info[model.replace("/", "-")].preds().to_pandas())


def transformation_example_1():
    col1, col2 = st.beta_columns(2)
    dp = display_dp(col1)


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    # st.title("RG Ops Demo")

    st.sidebar.title(":rocket: Robustness Gym\n " "## Cheat Sheet")

    st.sidebar.write("#### Installation")
    st.sidebar.code(
        """
$ pip install robustnessgym
# see setup.py for a list of optional packages
$ pip install robustnessgym[text]
        """,
        language="bash",
    )

    st.sidebar.write("#### Import convention")
    st.sidebar.code("import robustnessgym as rg")

    st.sidebar.write("### Selection")
    section = st.sidebar.radio(
        "Section",
        ["DataPanel", "Operation", "Subpopulation", "Demo"],
    )

    if section == "DataPanel":
        datapanel_page(size="all")

    elif section == "Operation":
        operation_page()

    elif section == "Subpopulation":
        # selection = st.sidebar.radio("Subpopulation Example", ["Subpopulation-1"])
        # if selection == "Subpopulation-1":
        subpopulation_example_1()

    elif section == "Transformation":
        selection = st.sidebar.selectbox("Transformation Example", ["Transformation-1"])
        if selection == "Transformation-1":
            transformation_example_1()

    elif section == "Demo":
        run_demo()
