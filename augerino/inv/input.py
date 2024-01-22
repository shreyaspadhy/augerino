import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from augerino.inv.preprocess import all_ops
from chex import PRNGKey
from clu import deterministic_data, preprocess_spec


def get_data(
    config,
    rng: PRNGKey,
):
    train_rng, val_rng, test_rng = jax.random.split(rng, 3)

    if config.dataset == "aug_dsprites":
        pass
    else:
        dataset_builder = tfds.builder(config.dataset)
        dataset_builder.download_and_prepare()
        dataset_or_builder = dataset_builder

    local_batch_size = config.batch_size // jax.device_count()

    train_ds = deterministic_data.create_dataset(
        dataset_or_builder,
        split=tfds.split_for_jax_process(config.train_split),
        # This RNG key will be used to derive all randomness in shuffling, data
        # preprocessing etc.
        rng=train_rng,
        num_epochs=config.num_epochs,
        shuffle_buffer_size=config.get("shuffle_buffer_size", 1),
        # Depending on TPU/other runtime, local device count will be 8/1.
        batch_dims=[jax.local_device_count(), local_batch_size],
        repeat_after_batching=False,
        preprocess_fn=preprocess_spec.parse(
            spec=config.pp_train,
            available_ops=all_ops(),
        ),
        shuffle="loaded",
    )

    if config.dataset == "aug_dsprites":
        pass
    else:
        num_val_examples = dataset_builder.info.splits[config.val_split].num_examples
        cardinality = None
    # Compute how many batches we need to contain the entire val set.
    pad_up_to_batches = int(jnp.ceil(num_val_examples / config.batch_size))

    val_ds = deterministic_data.create_dataset(
        dataset_or_builder,
        split=tfds.split_for_jax_process(config.val_split),
        rng=val_rng,
        batch_dims=[jax.local_device_count(), local_batch_size],
        num_epochs=1,
        preprocess_fn=preprocess_spec.parse(
            spec=config.pp_eval,
            available_ops=all_ops(),
        ),
        # Pad with masked examples instead of dropping incomplete final batch.
        pad_up_to_batches=pad_up_to_batches,
        cardinality=cardinality,
        shuffle=False,
    )

    test_split = config.get("test_split", None)
    if test_split is None:
        return train_ds, val_ds, None

    num_test_examples = dataset_builder.info.splits[test_split].num_examples
    # Compute how many batches we need to contain the entire test set.
    pad_up_to_batches = int(jnp.ceil(num_test_examples / config.batch_size))

    test_ds = deterministic_data.create_dataset(
        dataset_builder,
        split=tfds.split_for_jax_process(test_split),
        rng=test_rng,
        batch_dims=[jax.local_device_count(), local_batch_size],
        num_epochs=1,
        preprocess_fn=preprocess_spec.parse(
            spec=config.pp_eval,
            available_ops=all_ops(),
        ),
        # Pad with masked examples instead of dropping incomplete final batch.
        pad_up_to_batches=pad_up_to_batches,
        shuffle=False,
    )

    return train_ds, val_ds, test_ds
