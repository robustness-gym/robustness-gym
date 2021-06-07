import inspect

import streamlit as st
import numpy as np
from robustnessgym import DataPanel


@st.cache
def load_boolq():
    return DataPanel.load_huggingface('boolq', split='validation')


@st.cache
def load_boolq_small():
    return DataPanel.load_huggingface('boolq', split='validation[:2]')


def load_datapanel(dataset='boolq', size='small'):
    if dataset == 'boolq':
        if size == 'small':
            return load_boolq_small()
        else:
            return load_boolq()


def format_code(code, output=None, outputs=None, columns=(0.45, 0.1, 0.45)):
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


def datapanel_page(dataset='boolq', size='small'):
    st.write('# DataPanels: Get your data into Robustness Gym')
    st.write(
        """
Load data into Robustness Gym using the `DataPanel`. For detailed info about 
the `DataPanel`, check out [Mosaic](https://github.com/robustness-gym/mosaic).
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
        outputs=(dp['passage'].streamlit(), str(type(dp['passage']))),
        columns=[0.30, 0.05, 0.65]
    )
    format_code(
        """
dp['answer']
type(dp['answer'])
        """,
        outputs=(dp['answer'].streamlit(), str(type(dp['answer']))),
        columns=[0.30, 0.05, 0.65]
    )
    format_code(
        """
# NumpyArrayColumn works like np.ndarray!
dp['answer'].mean()
dp['answer'].reshape(-1, 1, 1)
        """
        "",
        outputs=(dp['answer'].mean(), dp['answer'].reshape(-1, 1, 1).streamlit()),
        columns=[0.30, 0.05, 0.65]
    )
    format_code("dp[0]", dp[0], columns=[0.30, 0.05, 0.65])
    format_code("dp[:1]", dp[:1].streamlit(), columns=[0.30, 0.05, 0.65])
    format_code("dp.identifier", dp.identifier, columns=[0.30, 0.05, 0.65])
    format_code("dp.columns", dp.columns, columns=[0.30, 0.05, 0.65])
    format_code(
        "dp.map(lambda x: len(x['passage']))",
        dp.map(lambda x: len(x['passage'])).streamlit(),
        columns=[0.30, 0.05, 0.65],
    )
    format_code(
        """
dp.map(lambda x: {
    'passage_len': len(x['passage']),  
    'question_len': len(x['question']),
})
        """,
        dp.map(lambda x: {
            'passage_len': len(x['passage']),
            'question_len': len(x['question']),
        }).streamlit(),
        columns=[0.30, 0.05, 0.65],
    )
    format_code(
        "dp.filter(lambda x: len(x['passage']) > 600)",
        dp.filter(lambda x: len(x['passage']) > 600).streamlit(),
        columns=[0.30, 0.05, 0.65],
    )
    format_code(
        "dp.update(lambda x: {'passage_len': len(x['passage'])})",
        dp.update(lambda x: {'passage_len': len(x['passage'])}).streamlit(),
        columns=[0.30, 0.05, 0.65],
    )
    format_code(
        """
dp.update(lambda x: {
    'passage_len': len(x['passage']),  
    'question_len': len(x['question']),
})
        """,
        dp.update(lambda x: {
            'passage_len': len(x['passage']),
            'question_len': len(x['question']),
        }).streamlit(),
        columns=[0.30, 0.05, 0.65],
    )
    format_code("dp.to_pandas()", dp.to_pandas(), columns=[0.30, 0.05, 0.65])

    st.write('------')


def display_dp(dataset='boolq', size='small'):
    st.subheader('Load Data')
    code_col, stdout_col = st.beta_columns(2)
    with code_col:
        if dataset == 'boolq':
            if size == 'small':
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
    st.write('# Operations: Run common processing workflows in Robustness Gym')
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

    with st.beta_expander('Run the spaCy pipeline with `SpacyOp`'):
        spacy_example(dp)

    with st.beta_expander('Run the Stanza pipeline with `StanzaOp`'):
        stanza_example(dp)

    with st.beta_expander('Lazily run the TextBlob pipeline with `LazyTextBlobOp`'):
        lazy_textblob_example(dp)

    with st.beta_expander('Write a simple `Operation` to capitalize text'):
        custom_operation_example_1(dp)

    with st.beta_expander('Write an `Operation` that adds '
                          'multiple columns to capitalize and upper-case text'):
        custom_operation_example_2(dp)


def spacy_example(dp):

    with st.beta_container():
        st.header('Run Operation: `SpacyOp`')
        st.write(
            """
spaCy is a popular text processing library that provides tokenization, tagging 
and other capabilities. 
            """
        )
        code_col, stdout_col = st.beta_columns(2)

        with code_col:
            with st.echo():
                from robustnessgym import lookup
                from robustnessgym.ops import SpacyOp

                # Run the Spacy pipeline on the 'question' column of the dataset
                spacy = SpacyOp()
                dp = spacy(dp=dp, columns=['passage'])
                # adds a new column that is auto-named
                # "SpacyOp(lang=en_core_web_sm, neuralcoref=False, columns=['passage'])"

        with stdout_col:
            with st.echo():
                st.write(dp.column_names)

    st.write('------')

    with st.beta_container():
        code_col2, stdout_col2 = st.beta_columns(2)

        with code_col2:
            with st.echo():
                # Grab the Spacy column from the DataPanel using the lookup
                spacy_column = lookup(dp, spacy, ['passage'])

        with stdout_col2:
            with st.echo():
                st.write(spacy_column._repr_pandas_())

    with st.beta_container():
        st.subheader('Columns contain Cells')
        code_col3, stdout_col3 = st.beta_columns(2)
        with code_col3:
            with st.echo():
                # Each element in the column is a SpacyCell
                # that behaves like a Spacy Doc
                cell = spacy_column[1]

        with stdout_col3:
            with st.echo():
                # Print the cell
                st.write(cell)

            with st.echo():
                # Behaves like a Spacy Doc object: list(cell) outputs tokens
                st.write(list(cell))

            with st.echo():
                # Access other attributes normally
                st.write(cell.ents)


def stanza_example(dp):
    st.header('Run Stanza Workflow')
    col1, col2 = st.beta_columns(2)
    with col2:
        with st.echo():
            from robustnessgym import lookup
            from robustnessgym.ops import StanzaOp

            # Run the Stanza pipeline on the 'question' column of the dataset
            stanza = StanzaOp()
            dp = stanza(dp=dp, columns=['question'])
            # adds a new column that is auto-named "StanzaOp(columns=['question'])"

            # Grab the Stanza column from the DataPanel using the lookup
            stanza_column = lookup(dp, stanza, ['question'])
            st.write(stanza_column._repr_pandas_())

        st.subheader('Columns contain Cells')
        with st.echo():
            # Each element in the column is a StanzaCell
            cell = stanza_column[1]
            st.write("StanzaCell", str(type(cell)))

            # Behaves like a Stanza Document object
            st.write(cell.text)
            st.write(cell.entities)


def lazy_textblob_example(dp):
    st.header('Run TextBlob Workflow Lazily')
    col1, col2 = st.beta_columns(2)

    with col2:
        with st.echo():
            from robustnessgym import lookup
            from robustnessgym.ops import LazyTextBlobOp

            # Run the TextBlob pipeline on the 'passage' column of the dataset
            textblob = LazyTextBlobOp()
            dp = textblob(dp=dp, columns=['passage'])
            # adds a new column that is auto-named "LazyTextBlobOp(columns=['passage'])"

            # Grab the TextBlob column from the DataPanel using the lookup
            textblob_column = lookup(dp, textblob, ['passage'])
            st.write(textblob_column._repr_pandas_())

        st.subheader('Columns contain cells')
        with st.echo():
            # Each element in the column is a LazyTextBlobCell
            cell = textblob_column[1]
            st.write("LazyTextBlobCell", str(type(cell)))

            # Behaves like a TextBlob object
            st.write(cell)
            st.write(cell.sentences)


def custom_operation_example_1(dp):
    st.header('Run Custom Operation')
    col1, col2 = st.beta_columns(2)

    with col2:
        with st.echo():
            from robustnessgym import Operation, Id, lookup

            # A function that capitalizes text
            def capitalize(batch, columns):
                return [text.capitalize() for text in batch[columns[0]]]

            # Wrap in an Operation: `process_batch_fn` accepts functions that have
            # exactly 2 arguments: batch and columns, and returns a tuple of outputs
            op = Operation(
                identifier=Id('CapitalizeOp'),
                process_batch_fn=capitalize,
            )
            st.write(op)

        with st.echo():
            # Apply to a DataPanel
            dp = op(dp=dp, columns=['question'])

            # Look it up when you need it
            capitalized_text = lookup(dp, op, ['question'])
            st.write(capitalized_text._repr_pandas_())

            # It's just text
            st.write(capitalized_text[0])


def custom_operation_example_2(dp):
    st.header('Run Custom Operation')
    col1, col2 = st.beta_columns(2)

    with col2:
        with st.echo():
            from robustnessgym import Operation, Id

            # A function that capitalizes and upper-cases text: this will
            # be used to add two columns to the DataPanel
            def capitalize_and_upper(batch, columns):
                return [text.capitalize() for text in batch[columns[0]]], \
                       [text.upper() for text in batch[columns[0]]]

            # Wrap in an Operation: `process_batch_fn` accepts functions that have
            # exactly 2 arguments: batch and columns, and returns a tuple of outputs
            op = Operation(
                identifier=Id('ProcessingOp'),
                output_names=['capitalize', 'upper'],
                process_batch_fn=capitalize_and_upper,
            )
            st.write(op)

        with st.echo():
            from robustnessgym import lookup

            # Apply to a DataPanel
            dp = op(dp=dp, columns=['question'])

            # Adds 2 columns to the DataPanel
            st.write(dp.column_names)
            st.write(dp._repr_pandas_())

            # Look them up when you need them
            capitalized_text = lookup(dp, op, ['question'], 'capitalize')
            upper_text = lookup(dp, op, ['question'], 'upper')
            st.write(capitalized_text._repr_pandas_())
            st.write(upper_text._repr_pandas_())


def subpopulation_example_1():
    col1, col2 = st.beta_columns(2)
    dp = display_dp(col1, size='all')

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

            slices, membership = length_sp(dp=dp, columns=['passage'])
            # `slices` is a list of 2 DataPanels
            # `membership` is an np.ndarray of shape (n x 2)
            st.markdown('`slices`')
            st.write(slices)
            st.markdown('`membership`')
            st.write(membership)

        with st.echo():
            # Create a subpopulation that buckets examples based on length
            length_sp = ScoreSubpopulation(
                # mixture of percentile intervals and raw intervals
                intervals=[('0%', '10%'), ('10%', '20%'), (10, 20)],
                score_fn=length,
            )

            slices, membership = length_sp(dp=dp, columns=['passage'])
            st.write('`slices` sizes:', [len(sl) for sl in slices])
            st.write('`membership` shape:', membership.shape)

        st.write('### Updated intervals')
        with st.echo():
            # The percentile intervals are updated to
            # raw values after running the subpopulation
            st.write(length_sp.intervals)

        st.write('### Examples and average passage length for each slice')
        with st.echo():
            for sl in slices:
                # The slice identifier
                st.write(str(sl.identifier))
                # Look at the first 3 examples from the slice
                st.write(sl.head(3)._repr_pandas_())
                # Calculate the average length of the passages in the slice
                st.write(sl.map(
                    lambda x: length(x, ['passage']),
                    is_batched_fn=True
                ).mean())


def transformation_example_1():
    col1, col2 = st.beta_columns(2)
    dp = display_dp(col1)


if __name__ == '__main__':
    st.set_page_config(layout="wide")

    # st.title("RG Ops Demo")

    st.sidebar.title(
        ':rocket: Robustness Gym\n '
        '## Cheat Sheet'
    )

    st.sidebar.write('#### Installation')
    st.sidebar.code(
        """
$ pip install robustnessgym
# see setup.py for a list of optional packages
$ pip install robustnessgym[text]
        """,
        language="bash"
    )

    st.sidebar.write('#### Import convention')
    st.sidebar.code('import robustnessgym as rg')

    st.sidebar.write('### Selection')
    section = st.sidebar.radio(
        'Section',
        ['DataPanel', 'Operation', 'Subpopulation'],
    )

    if section == 'DataPanel':
        datapanel_page(size='all')

    elif section == 'Operation':
        operation_page()

    elif section == 'Subpopulation':
        selection = st.sidebar.radio(
            'Subpopulation Example',
            ['Subpopulation-1']
        )
        if selection == 'Subpopulation-1':
            subpopulation_example_1()

    elif section == 'Transformation':
        selection = st.sidebar.selectbox(
            'Transformation Example',
            ['Transformation-1']
        )
        if selection == 'Transformation-1':
            transformation_example_1()
