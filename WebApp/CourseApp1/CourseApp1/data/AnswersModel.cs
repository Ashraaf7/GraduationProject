using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace CourseApp1.data
{
    [MetadataType(typeof(AnswerMetadata))]
    public partial class Answer
    {

    }
    public class AnswerMetadata
    {
        public int AnswerId { get; set; }
        [Display(Name = "Answer")]
        public string Content { get; set; }
        public int QuestionId { get; set; }
        public string UserId { get; set; }
        public System.DateTime CreationDate { get; set; }
    }
}