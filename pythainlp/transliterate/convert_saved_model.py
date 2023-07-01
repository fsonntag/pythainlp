import tensorflow as tf
import numpy as np

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
    # with open("thai_w2p_tf_model.tflite", "wb") as f:
    #     f.write(tflite_model)

    interpreter = tf.lite.Interpreter("thai_w2p_tf_model.tflite")
    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    my_signature = interpreter.get_signature_runner()

    # my_signature is callable with input as arguments.
    word_tensor = tf.convert_to_tensor("โทรศัพท์")

    # output = my_signature(input_1=tf.constant(["โทรศัพท์"], shape=word_tensor.shape, dtype=tf.string))
    output = my_signature(input_1=np.array(["โทรศัพท์"]))
    # output = my_signature(input_1=word_tensor.numpy())
    # 'output' is dictionary with all outputs from the inference.
    # In this case we have single output 'result'.
    byte_output = output["output_1"]
    print(byte_output)
    print(bytes.decode(b"".join(byte_output)))
