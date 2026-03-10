---
name: deploy
description: Deploy autoresearch training to an EC2 GPU instance
disable-model-invocation: true
allowed-tools: Bash(aws *), AskUserQuestion
---

# Deploy autoresearch to EC2

## Step 1: Ask for AWS profile

Use AskUserQuestion to ask which AWS profile to use. Read `~/.aws/config` to list available profiles as options.

## Step 2: Find the AMI

Run this command with the chosen profile:

```bash
AWS_PROFILE=<profile> aws ec2 describe-images \
  --owners amazon \
  --filters \
    "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
    "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text
```

## Step 3: Launch the instance

Run this command using the AMI from step 2, the chosen profile, and the userdata script at `deploy/userdata.sh` in this repo:

```bash
AWS_PROFILE=<profile> aws ec2 run-instances \
  --image-id <ami-id> \
  --instance-type g6.xlarge \
  --key-name jack \
  --user-data file://deploy/userdata.sh \
  --query 'Instances[0].InstanceId' \
  --output text
```

## Step 4: Wait for public IP

The instance may not have a public IP immediately. Poll until it appears:

```bash
AWS_PROFILE=<profile> aws ec2 describe-instances \
  --instance-ids <instance-id> \
  --query 'Reservations[0].Instances[0].[PublicIpAddress,State.Name]' \
  --output text
```

Wait a few seconds between checks. Stop after 60 seconds if no IP appears and tell the user.

## Step 5: Print the SSH command

Once the instance is running and has a public IP, print:

```
Instance ID: <instance-id>
Public IP:   <public-ip>

SSH command:
  ssh -i ~/.ssh/jack.pem ubuntu@<public-ip>
```

Remind the user that userdata setup (apt-get, uv, git clone) takes a few minutes to complete after the instance starts.
