from magicarp.pipeline.best_prompts.best_prompts import BestPromptsPipeline

pipe = BestPromptsPipeline(
	"C:\\Users\\Kavin\\Documents\\GitHub\\magicarp-v2\\magicarp\\pipeline\\best_prompts\\prompts.csv",
	"C:\\Users\\Kavin\\Documents\\GitHub\\magicarp-v2\\magicarp\\pipeline\\best_prompts\\annotation.csv"
	)

print(len(pipe))
print(pipe[0])
# loader = pipe.create_loader(batch_size = 8)