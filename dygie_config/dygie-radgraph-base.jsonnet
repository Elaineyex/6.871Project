local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "bert-base-uncased",
  cuda_device: 0,
  data_paths: {
    train: "./data/dygie_train.json",
    validation: "./data/dygie_dev.json",
    test: "./data/dygie_test.json",
  },
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "relation",
  trainer +: {
    num_epochs: 30
  }
}
