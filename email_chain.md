Of course. Here is the entire email chain, cleaned of all duplicates and formatted in Markdown for clarity and preservation.

# Email Conversation: AI Project for ICD Detection

**Project:** Development of an AI model to identify Implantable Cardioverter-Defibrillators (ICDs) from chest X-ray images.

**Primary Participants:**
- **Linh Nguyen:** Medical Student, University of the Incarnate Word School of Osteopathic Medicine
- **Dr. Kal L Clark:** UT Health San Antonio
- **Keola Ching:** Developer
- **Brian Desalme:** Developer

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Friday, August 2, 2024 3:21 PM
**Subject:** Hello from Linh Nguyen

Dear Dr. Clark,

I hope this email finds you well. My name is Linh Nguyen, a second-year medical student at the University of the Incarnate Word School of Osteopathic Medicine. I’ve met with Dr. Rohweder at his office recently and expressed to him my interest in Radiology and Interventional radiology, as well as the UT Health San Antonio Radiology Program. Dr. Rohweder has introduced me to you for any research opportunities you may have.

Before entering medical school, I was an Xray technologist for 2 years and then MRI technologist for another 6 years. Now in medical school, my interest in Radiology only grows larger. Please let me know if you have any questions or concerns. Please allow me to attach here my CV for your review.

I’m very much looking forward to hearing from you and truly appreciate your patience and kindness. Have a blessed day, Dr. Clark!

Sincerest regards,

Linh Nguyen
OMS-2, Class of 2027
Radiology Interest Group President
Asian Pacific American Medical Students Association Vice President
University of the Incarnate Word School of Osteopathic Medicine

---

**From:** Clark, Kal L
**To:** Nguyen, Linh
**Date:** Friday, August 2, 2024 3:31 PM
**Subject:** RE: Hello from Linh Nguyen

Hi Linh, nice to meet you. Yes we are doing lots of stuff in radiology AI and economics.

Let’s chat

Please pick a time that is convenient for you here: https://calendly.com/kalclark/30min

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Friday, August 2, 2024 3:45 PM
**Subject:** Re: Hello from Linh Nguyen

Hi Dr. Clark!

Thank you very much for your fast response! I’ve picked next Tuesday at 1:30pm for the meeting session. I’m very much looking forward to meeting you!

Have a great weekend!

Sincerely,

Linh Nguyen
OMS-2, Class of 2027
Radiology Interest Group President
Asian Pacific American Medical Students Association Vice President
University of the Incarnate Word School of Osteopathic Medicine

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Saturday, August 10, 2024 9:34 AM
**Subject:** FW: Hello from Linh Nguyen

Good morning, Dr. Clark!

I hope this email finds you well! Attached is my Briefing draft. It’s a little longer than one page as we expected it. Please let me know how you think and adjust if needed, as well as when you want a final paper with all the references included.

Thank you very much for giving me this wonderful opportunity to work with you and learn from you. Thank you for being patient with me, too!

Have a great weekend!

Sincerest regards,
Linh

---

**From:** Clark, Kal L
**To:** Nguyen, Linh
**Date:** Wednesday, August 21, 2024 2:19 PM
**Subject:** RE: Hello from Linh Nguyen

Hi Linh, sorry for delay, I like this let’s do it.

Practically what we would do is:
1. There is a public repository of images which contain images for patients who have ICD.
2. Let’s have you take a look at a sample of 5 and see how you would go about figuring out their ICD type.
3. We will also have the ability to query clinical notes in the near future which may give you additional information.
4. Once you feel comfortable in “labeling” these images we can move forward with an algorithm which does this automatically.

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Wednesday, August 21, 2024 4:33 PM
**Subject:** Re: Hello from Linh Nguyen

Good afternoon, Dr. Clark!

Thank you very much for your email! And I totally understand how busy it can get for you. I can always send a friendly reminder time to time.

It sounds great! I’m very much looking forward to doing it. This is really my first time ever! Feeling excited and a little nervous as well.

Have a great evening!

Sincerely,
Linh

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Saturday, September 14, 2024 4:15 PM
**Subject:** Re: Hello from Linh Nguyen

Good afternoon, Dr. Clark!

It was great to see you in person at the symposium today. Thank you very much for making it happen for all of us! The food was great, too!

I’m very much looking forward to working with you more and moving forward on our project. I’m free most of the afternoons of the week, so please let me know your availability.

Have a wonderful weekend!

Respectfully,
Linh

---

**From:** Clark, Kal L
**To:** Nguyen, Linh
**Date:** Monday, September 16, 2024 4:17 PM
**Subject:** RE: Hello from Linh Nguyen

Hi Linh,

I’m sharing with you a folder of images from the MIMIC dataset.
https://drive.google.com/drive/u/0/folders/14pJqn1QUMiZcTMp5unwZUr_sLjxp3_96

What I would recommend doing:
1. Review some of the images and see how you would characterize them.
2. Then I think what will be most exciting and learning filled for you is to go to https://roboflow.com/annotate and try to build a model yourself.
3. Perhaps start with trying to predict where on the image the pacemaker or defibrillator is.
4. Then move towards predicting the type of pacemaker defibrillator.
5. There are lots of Youtube videos on how to do this.
6. If you get stuck or don’t really want to do the coding part then at the minimum we’ll need to assess how accurately you can “label” or differentiate between different kinds of pacemakers or ICDs. Review maybe 50 randomly and see if you notice any patterns or if this is possible.

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Wednesday, September 18, 2024 8:28 PM
**Subject:** Re: Hello from Linh Nguyen

Good evening, Dr. Clark!

Thank you for your email and guidance!

Looking through the ICD images, I do recognize that some patterns are a little easier to stand out. I have 2 questions if you don’t mind:
1. In the Image Folder: I notice that even though the name of each image file irregular, seems like random, but they are actually in “order” from “a” to “f”. Does it mean something, or just me overdoing it? 😊
2. There is another folder name “Transcripts”, which are reports. Is there any image correlated to these reports? (Or the transcripts are just something else?)

I’m watching a couple of videos about Roboflow. I really hope we can carry this project as far as we can.

This is what I’m planning to do (typing out loud, so please correct me if I’m not right):
1. Categorize some big US companies that make ICDs and their products, plus xray images showing how they look. (Simply using google) Getting familiar with the patterns myself.
2. Download the images that look good: mostly AP/PA, clear, ICDs and leads are within the xray images (without getting cut).
3. Upload these images to Roboflow and start labeling them.

Please give me a week. I also add you to this project by adding your email. Thank you very much, Dr. Clark!

Have a wonderful evening!

Sincerely,
Linh

---

**From:** Clark, Kal L
**To:** Nguyen, Linh
**Date:** Thursday, September 19, 2024 9:07 PM
**Subject:** RE: Hello from Linh Nguyen

Hi Linh, great to hear from you.

- **Image folder:** these are “UUID” which is a universally unique code which is convenient for medical imaging. Doesn’t mean anything fancy.
- **Transcripts:** these are named after the study id
- **Metadata csv file:** this is the “dictionary” of how you find the transcript that goes with the image. Basically look for the dicomid that goes with the subject id.
- **Project next steps:** Yes on the Roboflow plan. That sounds good to me. I think a technologist will do well with ICD versus pacemaker, the secret will be any additional info we can glean from the image that maybe tech is not familiar with (i.e. specific brand etc.)

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Monday, September 23, 2024 7:30 PM
**Subject:** Re: Hello from Linh Nguyen

Good evening, Dr. Clark!

From your previous email: “any additional info we can glean from the image that maybe tech is not familiar with (i.e. specific brand etc.)” => Definitely. This is our target. I’d like to define the brand and the model (if possible). Then, we can either link the result to a “Manual website” of the brand mentioning about MRI safety and conditional info, or giving the safety result straight.

I’m working on Roboflow again today and I believe I’ve hit a roadblock. The couple of tutor videos I’ve found are not so helpful for me…

Please let me explain what I understand and have been doing:
1. To be able to teach AI to recognize a specific ICD, I need to give to it an “image” + a “name” (label) => it means giving it a specific brand/model and how that model looks, so the AI can “learn” from what I teach it, and when it “sees” a similar one again, it can recognize.
2. As for images, I’ve been provided by the file you gave.
3. As for the specific name (brand/model), I don’t have a reference.
4. So I went online and found these two websites (even though there are only several brand/models):
   - https://www.bostonscientific.com/content/dam/bostonscientific/quality/education-
   - https://thoracickey.com/imaging-of-implantable-devices-2/
5. Then, I’ve taken screenshots of these ICDs (some in there are pacemakers), tried to upload them to Roboflow => purpose: to “teach” the AI what these models are.
6. Then, find similar ICDs from your file for the AI to practice recognizing it.

Here are my roadblocks:
1. Roboflow is quite challenging so far. I’ve asked friends around school if anyone are familiar with it, but no one yet => If you are good at it, would you please tutor or show me around when you have some free time?
2. The ICD images from those 2 websites above are limited resources. I’ve not been able to find more yet. => Any suggestions for more references, sir?

Overall, what do you think about where I’m at right now? Is there any understanding I have about how to do this project heading into an incorrect direction?

Thank you very much, Dr. Clark, for being patient with me! Have a great evening, sir!

Respectfully,
Linh

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Tuesday, October 1, 2024 9:47 AM
**Subject:** FW: Hello from Linh Nguyen

Good morning, Dr. Clark!

I hope this email finds you well! I’d love to check in with you again for the project. Please let me know your opinion, and if time is allowed, please let me know if you prefer to have a small Zoom session to discuss about what we can move forward with. I’m looking forward to discussing it with you.

Thank you very much, Dr. Clark! Have a great day!

Sincerely,
Linh

---

**From:** Clark, Kal L
**To:** Nguyen, Linh
**Date:** Wednesday, October 2, 2024 3:37 AM
**Subject:** RE: Hello from Linh Nguyen

Hi Linh,

Check out the Roboflow universe for some sample projects to see how people are labeling. Here is one where they are labeling ICD: https://universe.roboflow.com/chequeautodataentry/ge-healthcare-hackathon

Yes, you’ve got the idea correct. “Label”/tag images. Train a model to correctly take as an input an image and then output the correct tag.

Have you seen: https://blog.roboflow.com/getting-started-with-roboflow/

I think between that, the Roboflow Universe to see sample projects, and then ChatGPT you should be able to get it.

Give those a shot and then we can meet. More importantly IMO is how you enable/accelerate yourself to quickly acquire knowledge using AI tools (i.e. ChatGPT).

Please pick a time that is convenient for you here: https://calendly.com/kalclark/quickchat

---

**From:** Nguyen, Linh
**To:** Dang, Tri
**Date:** Wednesday, October 2, 2024 8:48 AM
**Subject:** Fw: Hello from Linh Nguyen

*(Note: Linh forwards the chain to Tri Dang.)*

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Sunday, November 10, 2024 10:40 PM
**Subject:** Re: Hello from Linh Nguyen

Dear Dr. Clark,

I’d like to update with you where I am. I tried to install myself Python and Pytorch, then using ChatGPT and Perplexity to teach me how to do. However, I got several hiccups. My computer for some reasons did not allow to fully download and use. I looked around and was introduced to a friend’s brother (Cody) who is in coding school. We worked together Friday evening to troubleshoot and set it up, where I have Python now. I’m waiting see him again on Thursday to download the rest, including PyTorch and a coding program. During the meeting, I did explain to him that I would use ChatGPT to help me write code and paste it over (the way that you explained to me).

Today, he suggested: we could utilize Amazon Rekognition and build out an application with AWS. He says it will save us having to build out the machine learning aspect and will be easily scalable.

What do you think about his point?

I have not had an opportunity to build an App like the way you have been doing yet, so I’m excited to try out. As for what Cody says, I can’t picture it out yet either. I’d love to hear from you.

Thank you, Dr. Clark!

Respectfully,
Linh

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Tuesday, November 12, 2024 6:08 PM
**Subject:** Re: Hello from Linh Nguyen

Good evening, Dr. Clark!

I hope all is well for you. When you have some moment, would you please share with me your thoughts on how we should proceed (the email below)? If you think we should stick to our previous plan (just using Python and PyTorch/Cursor to write the code), we will do that!

For what Cody suggests, I just don’t fully understand. It does sound convenient.

Thank you so much, Dr. Clark! Have a great evening!

Sincerely,
Linh

P/S: I have not scheduled a meeting with you this week because I don’t have significant progress yet. But please let me know if you prefer to meet up. Thank you, Dr. Clark!

---

**From:** Clark, Kal L
**To:** Nguyen, Linh
**Date:** Thursday, November 14, 2024 4:33 AM
**Subject:** RE: Hello from Linh Nguyen

Hi Linh,

I think you should start with Google Colab https://colab.research.google.com/ this will create a coding environment for you to use and reduce the initial startup headaches.

Click the Gemini tab and it will provide some AI assistance.

When you’re not working with that then watch videos on:
- Vercel v0
- Bolt.new
- Cursor.com
- Replit

Those 4 are popular prototyping tools that you can graduate to after you get your feet wet with colab.

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Thursday, November 14, 2024 7:58 AM
**Subject:** Re: Hello from Linh Nguyen

Good morning, Dr. Clark!

Thank you for your reply! I’ll get to work with it soon. Have a wonderful day!

Sincerely,
Linh

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Tuesday, November 19, 2024 3:22 PM
**Subject:** Re: Hello from Linh Nguyen

Good afternoon, Dr. Clark!

I hope this email finds you well! I’d like to update to you on our project progression. I’ve been on Google Colab and started writing codes, upload the file from Roboflow and let it run. I’ll keep you updated when having significant progress, probably then end of this week or early next week, so we can meet up if you are available.

Have a great afternoon, Dr. Clark!

Sincerely,
Linh

---

**From:** Clark, Kal L
**To:** Nguyen, Linh
**Date:** Tuesday, November 19, 2024 3:55 PM
**Subject:** RE: Hello from Linh Nguyen

Awesome, great news.

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Monday, December 16, 2024 10:35 AM
**Subject:** Re: Hello from Linh Nguyen

Good morning, Dr. Clark!

I hope this email finds you well. Please let me know if you are able to meet up for 15 minutes. I went ahead and schedule a meeting with you at 1:30pm this afternoon. If you have a specific time that works for you, please let me know. Have a great day!

Sincerely,
Linh Nguyen

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Thursday, January 2, 2025 11:02 PM
**Subject:** Re: Hello from Linh Nguyen

Happy New Year, Dr. Clark!

How are you? I wanted to update you on my progress so far and see when you might be available to meet and discuss how we can move forward. Please let me know your availability. Thank you, Dr. Clark!

Sincerely,
Linh

*[A screenshot of the Training result was attached here.]*

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Tuesday, January 7, 2025 7:13 PM
**Subject:** Re: Hello from Linh Nguyen

Good evening, Dr. Clark,

I hope this message finds you well! I would love the opportunity to meet with you to discuss the progress of our project. I’d like to update you on where we currently stand and seek your advice on the best direction moving forward.

Please let me know your availability, and I’ll book a time through your calendar website. I’m very much looking forward to our discussion.

Thank you so much for your guidance and support!

Sincerely,
Linh

---

**From:** Clark, Kal L
**To:** Nguyen, Linh
**Date:** Wednesday, January 8, 2025 10:33 AM
**Subject:** RE: Hello from Linh Nguyen

Hi Linh, those training metrics look good. Great, yes let’s meet.

Please pick a time that is convenient for you here: https://calendly.com/kalclark/quickchat

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Wednesday, January 8, 2025 11:30 AM
**Subject:** Re: Hello from Linh Nguyen

Hi Dr. Clark!

I’ve chosen this Friday at 1pm. Please let me know if that time works for you. I’m looking forward to seeing you!

Sincerely,
Linh

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Friday, January 10, 2025 1:37 PM
**Subject:** FW: Hello from Linh Nguyen

Hi Dr. Clark!

Thank you very much for the meeting. I thought I’d hit the recording video button, but it did not take it. ☹ Unfortunately I could not record what you showed me.

Here I’m just list out a few things, please help me correct it:
1. Cursor with added Chat gpt
2. Please help me write down the term “…”/website => to complete the project
3. Gethub : get on it and ask a programmer to …
4. Go to facilities/tech/radiologist to invite them to try out the website/algorithm

Thank you, Dr. Clark! Have a great day!

Sincerely,
Linh

---

**From:** Clark, Kal L
**To:** Nguyen, Linh
**Date:** Friday, January 10, 2025 2:44 PM
**Subject:** RE: Hello from Linh Nguyen

No prob.

- Huggingface
- Github
- Cursor has built-in: would recommend you watch some videos on X or Youtube of people doing a tutorial, it’s very popular now

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Sunday, March 23, 2025 11:09 PM
**Subject:** Re: Hello from Linh Nguyen

Dear Dr. Clark,

How have you been? I hope all is well for you and yours.

I’ve finished my last Unit (Neuro) of the first 2 years, took the final exams, and now am in the dedicated study time for Boards (COMLEX 1 and Step 1), which I will take at the beginning of May. Our project will be continued right after that, and hopefully sometime in the summer, it can be done, at least to the point that we discussed last time, where we will have a little webpage that can do some device recognition!

Thank you very much for your patience and support, Dr. Clark! Have a great evening!

Respectfully,
Linh

---

**From:** Clark, Kal L
**To:** Nguyen, Linh
**Date:** Monday, March 24, 2025 7:53 PM
**Subject:** Re: Hello from Linh Nguyen

Sounds good Linh, best of luck and talk to you soon.

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Tuesday, June 24, 2025 11:12 AM
**Subject:** Re: Hello from Linh Nguyen

Dear Dr. Clark,

I hope this email finds you well! How have you been?

A little update on our project: The past 1.5 months, I’ve been trying to get the website done (with the help of another friend). We’ve hit an obstacle and not able to move forward far enough. On Github, we’ve created a site, a submission/upload image area, but cannot go from here to generate the information/giving the answer for the uploaded image yet. (And now my friend has his knee surgery, so we have to postpone a little bit more). I thought we might be able to complete it before end of June, but I don’t think we can.

As for boards, because it’s just Pass/Fail, I took both Step 1 and Comlex 1 (for DO students) early, in beginning of April, 3 days in between, and got the results back at the end of April with passes. Since then, I’ve been trying to complete research projects and studying for Step 2.

I hope you are having a great summer with your loved ones.

Have a great day, Dr. Clark!

Sincerely,
Linh

---

**From:** Nguyen, Linh
**To:** Clark, Kal L
**Date:** Saturday, September 13, 2025 4:28 PM
**Subject:** FW: Hello from Linh Nguyen

Good afternoon, Dr. Clark,

It was great to see you again today. Below is the email I updated you in June about the status of the project.

It would be wonderful to have some technologist on board for the last part: making the website! Please let me know if there is something I can do.

Have a great weekend!

Sincerely,
Linh Nguyen RT(R)(MR)
OMS-3, Class of 2027
University of the Incarnate Word School of Osteopathic Medicine

---

**From:** Clark, Kal L
**To:** Nguyen, Linh; Ching, Keola; brian.desalme
**Date:** Saturday, September 13, 2025 6:30 PM
**Subject:** Re: Hello from Linh Nguyen

Hi Linh,

Adding on Brian and Keola here.

Linh has a project ready and just needs help on the technologist angle with training and hosting the model for demo purposes.

---

**From:** Nguyen, Linh
**To:** Clark, Kal L; Ching, Keola; brian.desalme
**Date:** Sunday, September 14, 2025 6:52 PM
**Subject:** Re: Hello from Linh Nguyen

Thank you very much, Dr. Clark!

Best,
Linh

---

**From:** Ching, Keola
**To:** Nguyen, Linh; Clark, Kal L; brian.desalme
**Date:** Wednesday, September 17, 2025 11:29 AM
**Subject:** Re: Hello from Linh Nguyen

Hi Linh!

Just caught up on the email chain, sounds like a very cool project! I'd be happy to help you turn this into a functional demo that people can upload a scan to.

What I need from you:
1. The `.pt` file for the model
2. A `requirements.txt` file for the model, alternatively the source code you used to create the model works too

I'm @DataDoc1 on github if it's easier to share a repo.

Best,
Keola Ching

---

**From:** Nguyen, Linh
**To:** Ching, Keola; brian.desalme
**Date:** Wednesday, September 17, 2025 5:10 PM
**Subject:** Re: Hello from Linh Nguyen

Hi Brian and Keola,

I’m sorry for this late response. I’m still looking for the `.pt` file and `.txt` file. Do you mind if we can zoom sometime? And we connect directly.

Thank you very much!

Sorry, Brian, that until now to get back to you. I'm in IM rotation at UH, and can’t get to the files as I don’t know them so well.

Best,
Linh

---

**From:** Ching, Keola
**To:** Nguyen, Linh; brian.desalme
**Date:** Saturday, September 20, 2025 6:11 PM
**Subject:** Re: Hello from Linh Nguyen

Sure, I'm in the OB ED at UH Mon-Thurs. Should be able to hop out at any time for a meeting. Let me know what time works best.

Hosting the model should only take a couple days once you have the files or source code.

Keola

---

**From:** Nguyen, Linh
**To:** Ching, Keola; brian.desalme
**Date:** Monday, September 22, 2025 5:26 AM
**Subject:** Re: Hello from Linh Nguyen

Good morning!

This week will be my last week at UH. Around 1-2pm/3pm usually I have some time. Let’s play by ear if we can make it today or other days. Otherwise we can always meet by Zoom.

My cell: 832-290-9325 for convenience.

Best,
Linh

---

**From:** Ching, Keola
**To:** Nguyen, Linh; brian.desalme; Clark, Kal L
**Date:** Tuesday, September 23, 2025 9:11 PM
**Subject:** Re: Hello from Linh Nguyen

Hi Linh,

Was nice to meet you yesterday. Here's a demo that predicts ICD type from a CXR. If you want any changes to the site, just describe the changes in an email and I can make it happen.

[ICD Detector - a Hugging Face Space by keola1001](https://huggingface.co/spaces/keola1001/icd-detector)
> Detect an ICD from a Chest XR

Cheers,
Keola

---

**From:** Nguyen, Linh
**To:** Ching, Keola; brian.desalme; Clark, Kal L
**Date:** Friday, September 26, 2025 9:01 AM
**Subject:** Re: Hello from Linh Nguyen

Hi Keola,

It’s very cool and I love the demo. Thank you very much. I’ve been playing with it here and there to test the accuracy. I’d like to chat with you more of how we can improve it. I’ll send texts!

Have a great day!

Best,
Linh

---

**From:** Clark, Kal L
**To:** Nguyen, Linh; Ching, Keola; brian.desalme
**Date:** *[Implied Friday, September 26, 2025]*
**Subject:** RE: Hello from Linh Nguyen

Nice