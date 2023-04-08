
from detector import *
from attacks.craft_adversarial_img import *


TEST_SMAPLES=1000
def main(detect_type, data_loader, models_folder, threshold, sigma, attack_type,
         device, data_type):
    ## mnist
    alpha = 0.05
    beta = 0.05
    detector = Detector(threshold=threshold, sigma=sigma, beta=beta, alpha=alpha, models_folder=models_folder,

                        max_mutated_numbers=100, device=device, data_type=data_type)
    adv_success = 0
    progress = 0
    avg_mutated_used = 0
    totalSamples = len(data_loader)
    if detect_type == 'adv':
        for img, true_label, adv_label in data_loader:
            rst, success_mutated, total_mutated = detector.detect(img, origi_label=adv_label)
            if rst:
                adv_success += 1
            avg_mutated_used += total_mutated
            progress += 1
            sys.stdout.write('\r Processed:%.2f %%' % (100.*progress/totalSamples))
            sys.stdout.flush()
    else:
        for img, true_label in data_loader:
            rst, success_mutated, total_mutated = detector.detect(img, origi_label=true_label)
            if rst:
                adv_success += 1
            progress += 1
            sys.stdout.write('\r Processed:%.2f' % (100. * progress / totalSamples))
            sys.stdout.flush()
            avg_mutated_used += total_mutated
    avg_mutated_used = avg_mutated_used * 1. / len(data_loader.dataset)

    total_data = len(data_loader.dataset)
    if detect_type == 'adv':
        logging.info(
            '{},{}-Adv Accuracy:{}/{},{:.4f},,avg_mutated_used:{:.4f}'.format(models_folder, attack_type, adv_success,
                                                                              total_data,
                                                                              adv_success * 1. / total_data,
                                                                              avg_mutated_used))
        avg_accuracy = adv_success * 1. / len(data_loader.dataset)
    else:
        logging.info(
            '{},Normal Accuracy:{}/{},{:.4f},,avg_mutated_used:{:.4f}'.format(models_folder, total_data - adv_success,
                                                                              total_data,
                                                                              1 - adv_success * 1. / total_data,
                                                                              avg_mutated_used))
        avg_accuracy = 1 - adv_success * 1. / total_data

    return avg_accuracy, avg_mutated_used

def show_progress(**kwargs):
    sys.stdout.write('\r Processed:%d' % (kwargs['progress']))
    sys.stdout.flush()


def get_data_loader(data_path, is_adv_data, data_type):
    if data_type == DATA_MNIST:
        img_mode = 'L'
        normalize = normalize_mnist
    elif data_type == DATA_svhn:
        normalize = normalize_svhn
        img_mode = None
    else:
        img_mode = None
        normalize = normalize_cifar10

    if is_adv_data:
        tf = transforms.Compose([transforms.ToTensor(), normalize])
        dataset = MyDataset(root=data_path, transform=tf, img_mode=img_mode, max_size=TEST_SMAPLES)  # mnist
        dataloader = DataLoader(dataset=dataset)
    else:
        dataset, channel = load_data_set(data_type, data_path, False)
        random_indcies = np.arange(10000)
        np.random.seed(random_seed)
        np.random.shuffle(random_indcies)
        random_indcies = random_indcies[:TEST_SMAPLES]
        data = datasetMutiIndx(dataset, random_indcies)
        dataloader = DataLoader(dataset=data)
    return dataloader


def get_wrong_label_data_loader(data_path, seed_model, data_type,device):
    dataset, channel = load_data_set(data_type, data_path, False)
    dataloader = DataLoader(dataset=dataset)
    wrong_labeled = samples_filter(seed_model, dataloader, return_type='adv', name='seed model',device=device)
    data = datasetMutiIndx(dataset, [idx for idx, _, _ in wrong_labeled][:TEST_SMAPLES])
    wrong_labels = [wrong_label for idx, true_label, wrong_label in wrong_labeled[:TEST_SMAPLES]]
    data = TensorDataset(data.tensors[0], data.tensors[1], torch.LongTensor(wrong_labels))
    return DataLoader(dataset=data)

def get_threshold_relax(threshold, extend_scale, relax_scale):
    return threshold * extend_scale, threshold * relax_scale




def run():

    threshold, sigma = get_threshold_relax(args.threshold, 1.0, 0.1)
    device = args.multigpu[0]
    if args.set == 'cifar10':
        data_type = DATA_CIFAR10
        sourceDataPath = '../dataset/cifar10'
    else:
        data_type = DATA_svhn
        sourceDataPath = '../dataset/svhn'

    if args.testType == "normal":
        data_loader = get_data_loader(sourceDataPath, is_adv_data=False, data_type=data_type)
        avg_accuracy, avg_mutated_used = main('normal', data_loader, args.prunedModelsPath, threshold, sigma, 'normal',
                                              device=device, data_type=data_type)
    elif args.testType == "wl":
        model = get_model(args)
        # print(model)
        model = set_gpu(args, model)
        if args.pretrained:
            pretrained(args, model
                       )
        data_loader = get_wrong_label_data_loader(sourceDataPath, model, data_type,device=device)
        avg_accuracy, avg_mutated_used = main('adv', data_loader, args.prunedModelsPath, threshold, sigma,
                                              'wl', device=device,
                                              data_type=data_type)
    elif args.testType == "adv":
        data_loader = get_data_loader(args.testSamplesPath, is_adv_data=True, data_type=data_type)
        avg_accuracy, avg_mutated_used = main('adv', data_loader, args.prunedModelsPath, threshold, sigma,"adv",
                                                  device=device, data_type=data_type)
    else:
        raise Exception("Unsupported test type.")

    print("average accuracy:{}, average mutants used:{}".format(avg_accuracy,avg_mutated_used))


if __name__=="__main__":
    run()




