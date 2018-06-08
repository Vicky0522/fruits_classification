import time
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

# train
opt_train = TrainOptions().parse()
data_loader_train = CreateDataLoader(opt_train)
dataset_train = data_loader_train.load_data()
dataset_size_train = len(data_loader_train)
print('#training images = %d' % dataset_size_train)

# test
opt_test = TestOptions().parse()
opt_test.nThreads = 1   # test code only supports nThreads = 1
opt_test.batchSize = 1  # test code only supports batchSize = 1
opt_test.serial_batches = False  # no shuffle
opt_test.no_flip = True  # no flip
opt_test.how_many = 100
data_loader_test = CreateDataLoader(opt_test)
dataset_test = data_loader_test.load_data()
dataset_size_test = len(data_loader_test)
print('#test images = %d' % dataset_size_test)

model = create_model(opt_train)
visualizer = Visualizer(opt_train)
total_steps = 0

for epoch in range(opt_train.epoch_count, opt_train.niter + opt_train.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset_train):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt_train.batchSize
        epoch_iter += opt_train.batchSize
        model.set_input(data)
        model.optimize_parameters()

        #if total_steps % opt.display_freq == 0:
        #    save_result = total_steps % opt.update_html_freq == 0
        #    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt_train.print_freq == 0:
            # test cases
            for j ,data_test in enumerate(dataset_test):
                if j >= opt_test.how_many:
                    break
                model.set_input(data_test)
                model.test()

            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt_train.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt_train.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size_train, opt_train, errors)


        if total_steps % opt_train.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt_train.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt_train.niter + opt_train.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
