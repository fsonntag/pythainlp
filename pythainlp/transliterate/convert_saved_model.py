import tensorflow as tf
import numpy as np

from pythainlp.transliterate.w2p_tf import _load_vocab

if __name__ == "__main__":
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(
        "thai_w2p_tf_model"
    )  # path to the SavedModel directory
    # converter._experimental_lower_tensor_list_ops = False
    # converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()

    # Save the model.
    with open("thai_w2p_tf_model.tflite", "wb") as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter("thai_w2p_tf_model.tflite")
    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    my_signature = interpreter.get_signature_runner()

    # output = my_signature(input_1=tf.constant(["โทรศัพท์"], shape=word_tensor.shape, dtype=tf.string))

    word = "โทรศัพท์"
    g2idx = _load_vocab()[0]
    chars = list(word) + ["</s>"]
    x = [g2idx.get(char, g2idx["<unk>"]) for char in chars]

    output = my_signature(input_1=np.array(x, dtype=np.int32))
    byte_output = output["output_1"]
    print(len(byte_output))
    print(byte_output)
    print(bytes.decode(b"".join(byte_output)))
