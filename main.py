from data.compression_experiment import CompressionExperiment


def main():

    # Các cấu hình thực nghiệm (k_Y, k_C)
    configs = [(30, 10), (60, 20), (120, 40), (240, 80), (480, 160)]

    experiment = CompressionExperiment(image_path="input.jpg", configs=configs)

    experiment.run()


if __name__ == "__main__":
    main()
