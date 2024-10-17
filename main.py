from sammo.mutators import *
from sammo.runners import AzureChat, AzureEmbedding
from sammo import search_op
from sammo.instructions import *
from sammo.components import *
from sammo.dataformatters import QuestionAnswerFormatter
from sammo.search import EnumerativeSearch
import sammo.store

# Import custom modules prepared for use case
from data_preprocess import ConfigLoader
from metric_calculation import F1ScoreCalculator

logger = sammo.setup_logger(log_prompts_to_file=True)
_ = sammo.setup_logger("WARNING")

# Load API configurations
gpt_config_loader = ConfigLoader("gpt_config.json")
gpt_api_config = gpt_config_loader.get_config()

embed_config_loader = ConfigLoader("embedding_config.json")
embed_api_config = embed_config_loader.get_config()

passage = """日本には、「学習まんが」という本がある。歴史や経済など、学習する内容を文字ではなく、まんがで示したものだ。文字だけの本を読むのは苦手で勉強する気になれないという子供向けに作られたものが多いが、大人向けの「学習まんが」もある。
まんがとはいっても、学習まんがの場合、その内容は専門家がきちんとチェックしていて、しっかりと学べるようになっている。しかし、大人の場合、教えるほうも学ぶほうも、まんがで勉強することには、否定的な人もいるようだ。そういう人は、心のどこかで「文字で勉強するほうがレベルが高い」と考えていないだろうか。最終的に、その知識が身に付けばいいのだから、もんがのほうがよく理解できて覚えやすいなら、それはその人にとってよい学習方法と言えるだろう。学ぶ方法にレベルの高い、低いは関係ない。"""

question_1 = "「学習まんが」について、この文章で言っていることと違うのはどれか。"
answer_1 = "まんがで勉強する大人はいないので、大人向けのものはない。"
question_2 = "まんがで勉強することについて、この文章と合っているのはどれか。"
answer_2 = "大人中には、まんがで勉強することはよくない、と考える人がいる。"
question_3 = "まんがで学ぶことと文字で学ぶことについて、何と言っているか。"
answer_3 = "まんがでも文字でも、その人にとって理解しやすく覚えやすいのがよい方法だ。"

# Load the data to DataTable format
def load_data_qa():
    return DataTable(inputs=[passage])
d_context_ja = load_data_qa()

# Load the data to DataTable format
def load_data_train():
    return DataTable(inputs=[question_1, question_2],
                     outputs=[answer_1, answer_2],
                     constants={"instructions": "Answer the following questions based on the given passage."})
d_train_ja = load_data_train()

# Load the data to DataTable format
def load_data_test():
    return DataTable(inputs=[question_3],
                     outputs=[answer_3],
                     constants={"instructions": "Answer the following questions based on the given passage."})
d_test_ja = load_data_test()

class InitialCandidatesRAG:
    def __init__(self, dtrain, d_context_ja, embedding_runner):
        self.dtrain = dtrain
        self.d_context_ja = d_context_ja
        self._embedding_runner = embedding_runner

    def __call__(self, return_raw=False):
        orientation = search_op.one_of(["item", "kind"], name="orientation")
        example_formatter = search_op.one_of(
            [
                QuestionAnswerFormatter(
                    all_labels=self.dtrain.outputs.unique(), orient=orientation, attributes_processor=None
                ),
            ]
        )

        structure = [
            Section("Instruction", f"{self.dtrain.constants['instructions']}"),
            Section(
                "Examples",
                EmbeddingFewshotExamples(
                    self._embedding_runner,
                    self.d_context_ja,
                    budget="relative",
                ),
            ),
            Section(
                "Input",
                InputData(id_offset=len(self.dtrain)),
            ),
        ]
        instructions = MetaPrompt(structure, render_as="markdown", data_formatter=example_formatter)
        return Output(instructions.with_extractor("empty_result"), minibatch_size=1, on_error="empty_result")

runner = AzureChat(
    model_id="gpt-3.5-turbo-16k",
    api_config=gpt_api_config,  # Add key, endpoint, and deployment information
    timeout=30,
)

embedder = AzureEmbedding(
    model_id="text-embedding-3-small",
    api_config=embed_api_config,  # Add key, endpoint, and deployment information
    cache=None,
    timeout=30,
)

search_space = InitialCandidatesRAG(d_train_ja, d_context_ja, embedding_runner=embedder)

# Baseline
baseline_model = EnumerativeSearch(runner, search_space, F1ScoreCalculator.compute_f1_score, maximize=True, max_candidates=1)
baseline_model.fit_transform(d_train_ja)
