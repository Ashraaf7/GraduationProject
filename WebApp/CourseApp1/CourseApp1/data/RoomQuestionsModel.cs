using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace CourseApp1.data
{
    [MetadataType(typeof(RoomQuestionMetadat))]
    public partial class RoomQuestion
    {
    }
    public class RoomQuestionMetadat
    {
        public int QuestionId { get; set; }
        [Display(Name ="Question")]
        public string Content { get; set; }
        public int RoomId { get; set; }
        public string UserId { get; set; }
    }
    

}