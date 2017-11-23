import keras
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Dropout, Conv2D, Input, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

train_gen = ImageDataGenerator(
    horizontal_flip=True,
    rescale=1./255
)

test_gen = ImageDataGenerator(rescale=1./255)

batch_size = 4

train_generator = train_gen.flow_from_directory(
    '/root/sxwl-dataset/nas/dataset/keras_age_classify_dataset/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_gen.flow_from_directory(
    '/root/sxwl-dataset/nas/dataset/keras_age_classify_dataset/val',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001,
                            cooldown=0, min_lr=0)
if os.path.exists('age_resnet50.h5'):
    model = load_model('age_resnet50.h5')
else:
    # create the base pre-trained model
    input_tensor = Input(shape=(224, 224, 3))
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 8 classes
    predictions = Dense(8, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model = make_parallel(model, 3)
    # train the model on the new data for a few epochs

    model.fit_generator(train_generator,
                        steps_per_epoch=16626 / batch_size + 1,
                        epochs=30,
                        validation_data=test_generator,
                        validation_steps=3651 / batch_size + 1,
                        callbacks=[early_stopping, auto_lr])
    model.save('age_resnet50.h5')
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
cur_base_model = model#.layers[1]
for layer in cur_base_model.layers[:105]:
    layer.trainable = False
for layer in cur_base_model.layers[105:]:
    layer.trainable = True


# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
save_model = ModelCheckpoint('age_rennet50_tuned{epoch:02d}-{val_ctg_out_acc:.2f}.h5')
model.fit_generator(train_generator,
                    steps_per_epoch=16626 / batch_size + 1,
                    epochs=30,
                    validation_data=test_generator,
                    validation_steps=3651 / batch_size + 1,
                    callbacks=[early_stopping, auto_lr, save_model])  # otherwise the generator would loop indefinitely
model.save('age_rennet50_tuned.h5')