def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    if args.dataset_type == 'MS-COCO':
        args.do_bottleneck_head = False
        if args.th == None:
            args.th = 0.7
    elif args.dataset_type == 'OpenImages':
        args.do_bottleneck_head = True
        if args.th == None:
            args.th = 0.995
    return args
