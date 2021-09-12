def arch_eval_iter(choose_type='fromfile'):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  base_dir = 'search-EXP-20190917-114211'
  genotype_path = os.path.join(base_dir, 'results_of_7q/genotype')
  print(choose_type)
  if choose_type == 'fromfile':
    eval_dir = os.path.join(base_dir, 'eval_fromfile')
  else:
    eval_dir = os.path.join(base_dir, 'eval_sample/49')
  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
  val_acc = []
  val_loss = []
  genotype_names = [9,19,29,39,49]
  for name in genotype_names:
    if choose_type == 'fromfile':
      genotype_file = os.path.join(genotype_path, '%s.txt'%name)
      tmp_dict = json.load(open(genotype_file,'r'))
      genotype = genotypes.Genotype(**tmp_dict)
    elif choose_type == 'sample':
#      alpha_file = None
      alpha_file = os.path.join(base_dir, 'results_of_7q/alpha/%s.txt'%name)
      genotype = sample_genotype(alpha_file)
      # save genotype
      genotype_file = os.path.join(eval_dir, 'genotype_%s.txt'%name)
      with open(genotype_file, 'w') as f:
        json.dump(genotype._asdict(), f)
    else:
      raise(ValueError('No such choose_type: %s'%choose_type))
    print(genotype)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()
    # clear unused gpu memory
    torch.cuda.empty_cache() 

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )
  
  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  
    best_acc = 0;best_loss = 0
    for epoch in range(args.epochs):
      scheduler.step()
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
  
      train_acc, train_obj = train(train_queue, model, criterion, optimizer)
      logging.info('train_acc %f', train_acc)
  
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)
  
      if valid_acc > best_acc:
        try:
            last_model = os.path.join(eval_dir, 'weights_%s_%.3f.pt'%(name, best_acc))
            os.remove(last_model)
        except:
            pass
        utils.save(model, os.path.join(eval_dir, 'weights_%s_%.3f.pt'%(name, valid_acc)))
        best_acc = valid_acc
        best_loss = valid_obj

    val_acc.append(best_acc)
    val_loss.append(best_loss)
    print('Best: %f / %f'%(best_loss, best_acc))

  # save the results
  result_file = os.path.join(eval_dir, 'results.csv')
  with open(result_file, 'w') as f:
    writer = csv.writer(f)
    title = ['genotype_name', 'val_loss', 'val_acc']
    writer.writerow(title)
    for idx, loss in enumerate(val_loss):
      a = [genotype_names[idx], loss, val_acc[idx]]
      writer.writerow(a)

