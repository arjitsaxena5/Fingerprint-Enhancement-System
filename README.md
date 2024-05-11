# Fingerprint Enhancement System

This project focuses on enhancing fingerprint images using the RealESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) deep learning model. The goal is to improve the quality and resolution of fingerprint images, making them clearer and more suitable for further analysis or processing.

## Features

- Enhances fingerprint images using the RealESRGAN model
- Web application interface for easy upload and enhancement of fingerprint images
- Integration with AWS S3 for efficient storage and retrieval of fingerprint images
- AWS Lambda function for processing uploaded fingerprint images
- CloudWatch logging for monitoring and debugging

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/arjitsaxena5/Fingerprint-Enhancement-System/tree/master ```


2. Download the pre-trained RealESRGAN model checkpoint:
   - The checkpoint file (RealESRGAN_x4plus.pth) will be automatically downloaded if not available locally.

## Usage

1. Start the Flask web application:
   ```
   python app.py
   ```

2. Access the web application through a web browser:
   ```
   http://localhost:5000
   ```

3. Upload a fingerprint image using the provided interface.

4. Click the "Process Image" button to trigger the enhancement process.

5. View the enhanced fingerprint image and optionally download it.

## AWS Integration

This project integrates with AWS services for storage and processing of fingerprint images.

### AWS S3

- Fingerprint images are stored in an S3 bucket named “rawfingerprintdataset”.
- The bucket contains folders and images related to the fingerprint dataset.

### AWS Lambda

- The project includes an AWS Lambda function named "S3Trigger".
- The Lambda function is triggered when a new fingerprint image is uploaded to the specified S3 bucket.
- It processes the uploaded image by extracting relevant information (such as bucket name, object key) and performs further actions, such as logging or invoking other AWS services.

### AWS CloudWatch

- The project utilizes AWS CloudWatch for logging and monitoring.
- The CloudWatch log group captures log events associated with the execution of the Lambda function.
- The logs include details like the function ARN, runtime version, request and response payloads, and execution status.

## Future Enhancements

- Implement additional preprocessing techniques to handle variations in fingerprint image quality.
- Explore other deep learning architectures or models specifically designed for fingerprint enhancement.
- Incorporate additional features in the web application, such as user authentication and batch processing of multiple images.
- Optimize the AWS infrastructure setup to handle high-volume requests and ensure scalability and reliability.
- Conduct thorough testing and evaluation of the fingerprint enhancement system using a diverse dataset.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.


## Acknowledgements

- The RealESRGAN model is based on the research paper: [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
- The project utilizes the PyTorch deep learning framework and the Flask web framework.
